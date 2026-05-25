from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task


PROJECT_ROOT = Path(os.environ.get("FTD_EEG_PROJECT_ROOT", "/opt/airflow/project"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "ftd_ad_hc_final_artifacts")

FEATURE_DIR = Path(
    os.environ.get(
        "FTD_EEG_FEATURE_CACHE_DIR",
        str(PROJECT_ROOT / "airflow" / "cache" / "current_features"),
    )
)
EPOCH_DIR = Path(os.environ.get("FTD_EEG_EPOCH_DIR", str(PROJECT_ROOT / "Cleaned_Epochs")))
CONNECTIVITY_CACHE_DIR = Path(
    os.environ.get("FTD_EEG_CONNECTIVITY_WORK_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "connectivity"))
)
OUTPUT_DIR = PROJECT_ROOT / "notebook_outputs"
REPORT_DIR = OUTPUT_DIR / "dang_academic_report"
DAG_OUTPUT_DIR = OUTPUT_DIR / "airflow_final_pipeline"

BANDS = ("delta", "theta", "alpha", "beta", "gamma")
METRICS = (
    "cov",
    "corr",
    "xcov",
    "xcorr",
    "csd",
    "coh",
    "mi",
    "ecc",
    "aecov",
    "aecorr",
    "plv",
    "wplv",
)
EXPECTED_FEATURE_NAMES = tuple(f"{band}_{metric}" for band in BANDS for metric in METRICS)

PROBLEMS = {
    "ad_hc": "AD vs HC",
    "ftd_hc": "FTD vs HC",
    "ftd_ad": "FTD vs AD",
}

REQUIRED_FEATURE_FILES = [
    FEATURE_DIR / "labels.npy",
    FEATURE_DIR / "subject_ids.npy",
    FEATURE_DIR / "feature_metadata.csv",
    FEATURE_DIR / "all_feature_names.npy",
]
REQUIRED_FEATURE_FILES.extend(FEATURE_DIR / f"{feature_name}.npy" for feature_name in EXPECTED_FEATURE_NAMES)

REQUIRED_ARTIFACTS = [
    OUTPUT_DIR / "full_paper_60_v4_split_all_binary_losocv_metrics_summary.csv",
    OUTPUT_DIR / "full_paper_60_v4_split_paper_comparison_metrics.csv",
    REPORT_DIR / "IS252_Q22_Dang_sections_report.html",
    REPORT_DIR / "IS252_Q22_Dang_sections_report.odt",
]


@dag(
    dag_id="ftd_eeg_mlflow_final_pipeline",
    description="Validate final EEG classification outputs and log them to MLflow.",
    start_date=datetime(2026, 5, 25),
    schedule=None,
    catchup=False,
    tags=["eeg", "ftd", "ad", "mlflow", "final-report"],
)
def ftd_eeg_mlflow_final_pipeline():
    @task
    def ensure_feature_cache() -> dict[str, object]:
        missing_features = [str(path) for path in REQUIRED_FEATURE_FILES if not path.exists()]
        if not missing_features:
            return {
                "feature_cache_action": "reused_existing_cache",
                "feature_dir": str(FEATURE_DIR),
                "missing_before": [],
            }

        script_path = PROJECT_ROOT / "scripts" / "ensure_connectivity_cache.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing connectivity cache builder script: {script_path}")

        command = [
            sys.executable,
            str(script_path),
            "--input-dir",
            str(EPOCH_DIR),
            "--output-dir",
            str(FEATURE_DIR),
            "--cache-dir",
            str(CONNECTIVITY_CACHE_DIR),
        ]
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        result = json.loads(completed.stdout)
        result["missing_before"] = missing_features
        return result

    @task
    def validate_project_inputs(cache_status: dict[str, object]) -> dict[str, object]:
        missing = [str(path) for path in REQUIRED_FEATURE_FILES if not path.exists()]
        missing += [str(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required project artifacts:\n" + "\n".join(missing))

        return {
            "project_root": str(PROJECT_ROOT),
            "feature_dir": str(FEATURE_DIR),
            "feature_cache_action": cache_status.get("action")
            or cache_status.get("feature_cache_action", "unknown"),
            "output_dir": str(OUTPUT_DIR),
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
        }

    @task
    def summarize_feature_cache(project_info: dict[str, object]) -> dict[str, object]:
        import numpy as np
        import pandas as pd

        labels = np.load(FEATURE_DIR / "labels.npy", allow_pickle=True).astype(str)
        subject_ids = np.load(FEATURE_DIR / "subject_ids.npy", allow_pickle=True).astype(str)
        metadata = pd.read_csv(FEATURE_DIR / "feature_metadata.csv")

        summary = {
            "n_epochs": int(len(labels)),
            "n_subjects": int(len(set(subject_ids))),
            "n_feature_sets": int(len(metadata)),
            "label_counts": {label: int(count) for label, count in zip(*np.unique(labels, return_counts=True))},
            "feature_dir_size_bytes": sum(path.stat().st_size for path in FEATURE_DIR.glob("*") if path.is_file()),
            "feature_cache_action": project_info["feature_cache_action"],
        }

        DAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = DAG_OUTPUT_DIR / "feature_cache_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    @task
    def collect_final_metrics(project_info: dict[str, object]) -> dict[str, object]:
        import pandas as pd

        metrics_path = OUTPUT_DIR / "full_paper_60_v4_split_all_binary_losocv_metrics_summary.csv"
        comparison_path = OUTPUT_DIR / "full_paper_60_v4_split_paper_comparison_metrics.csv"

        metrics_df = pd.read_csv(metrics_path, index_col=0)
        comparison_df = pd.read_csv(comparison_path)

        metrics_records = metrics_df.reset_index().rename(columns={"index": "problem"}).to_dict(orient="records")
        comparison_records = comparison_df.to_dict(orient="records")

        payload = {
            "metrics_csv": str(metrics_path),
            "paper_comparison_csv": str(comparison_path),
            "feature_cache_action": project_info["feature_cache_action"],
            "metrics": metrics_records,
            "paper_comparison": comparison_records,
        }

        DAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (DAG_OUTPUT_DIR / "final_metrics_payload.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload

    @task
    def build_model_card(feature_summary: dict[str, object], metrics_payload: dict[str, object]) -> str:
        import pandas as pd

        DAG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame(metrics_payload["metrics"])

        lines = [
            "# EEG AD/FTD/HC Classification Final Model Card",
            "",
            "## Pipeline",
            "",
            "Final artifacts are generated from the closed notebook workflow. "
            "This Airflow DAG validates the feature cache and logs the fixed final outputs to MLflow.",
            "",
            "## Data And Features",
            "",
            f"- Feature directory: `{FEATURE_DIR}`",
            f"- Epochs: `{feature_summary['n_epochs']}`",
            f"- Subjects: `{feature_summary['n_subjects']}`",
            f"- Feature sets: `{feature_summary['n_feature_sets']}`",
            f"- Label counts: `{feature_summary['label_counts']}`",
            "",
            "## Model",
            "",
            "- Base classifier: FgMDM on functional connectivity matrices.",
            "- Meta-classifier: Logistic Regression Elastic Net.",
            "- Evaluation: leave-one-subject-out cross-validation with subject-level probability aggregation.",
            "",
            "## Final Metrics",
            "",
            metrics_df.to_markdown(index=False),
            "",
            "## Main Artifacts",
            "",
            f"- Final report HTML: `{REPORT_DIR / 'IS252_Q22_Dang_sections_report.html'}`",
            f"- Final report ODT: `{REPORT_DIR / 'IS252_Q22_Dang_sections_report.odt'}`",
            f"- Metrics summary: `{metrics_payload['metrics_csv']}`",
            f"- Paper comparison: `{metrics_payload['paper_comparison_csv']}`",
        ]

        model_card_path = DAG_OUTPUT_DIR / "model_card.md"
        model_card_path.write_text("\n".join(lines), encoding="utf-8")
        return str(model_card_path)

    @task
    def log_final_artifacts_to_mlflow(
        project_info: dict[str, object],
        feature_summary: dict[str, object],
        metrics_payload: dict[str, object],
        model_card_path: str,
    ) -> str:
        import mlflow
        import pandas as pd

        mlflow.set_tracking_uri(str(project_info["mlflow_tracking_uri"]))
        mlflow.set_experiment(str(project_info["mlflow_experiment_name"]))

        metrics_df = pd.DataFrame(metrics_payload["metrics"])
        run_name = "final_notebook_outputs_logged_by_airflow"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(
                {
                    "project_root": str(PROJECT_ROOT),
                    "feature_dir": str(FEATURE_DIR),
                    "feature_source": "precomputed_final_cache",
                    "feature_cache_action": project_info["feature_cache_action"],
                    "base_classifier": "FgMDM",
                    "meta_classifier": "LogisticRegressionElasticNet",
                    "evaluation": "LOSOCV_subject_level",
                    "n_epochs": feature_summary["n_epochs"],
                    "n_subjects": feature_summary["n_subjects"],
                    "n_feature_sets": feature_summary["n_feature_sets"],
                }
            )

            for _, row in metrics_df.iterrows():
                problem = str(row["problem"])
                if problem == "mean":
                    prefix = "mean"
                else:
                    prefix = problem
                for metric_name in ["roc_auc", "accuracy", "f1", "sensitivity", "specificity"]:
                    if metric_name in row and not pd.isna(row[metric_name]):
                        mlflow.log_metric(f"{prefix}_{metric_name}", float(row[metric_name]))

            artifacts = [
                Path(metrics_payload["metrics_csv"]),
                Path(metrics_payload["paper_comparison_csv"]),
                Path(model_card_path),
                REPORT_DIR / "IS252_Q22_Dang_sections_report.html",
                REPORT_DIR / "IS252_Q22_Dang_sections_report.odt",
                DAG_OUTPUT_DIR / "feature_cache_summary.json",
                DAG_OUTPUT_DIR / "final_metrics_payload.json",
            ]

            for problem in PROBLEMS:
                artifacts.extend(
                    [
                        OUTPUT_DIR / f"full_paper_60_v4_split_{problem}_losocv_metrics.csv",
                        OUTPUT_DIR / f"full_paper_60_v4_split_{problem}_losocv_subject_predictions.csv",
                        OUTPUT_DIR / f"full_paper_60_v4_split_{problem}_losocv_fold_artifacts.json",
                    ]
                )

            for artifact_path in artifacts:
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            return run.info.run_id

    cache_status = ensure_feature_cache()
    project_info = validate_project_inputs(cache_status)
    feature_summary = summarize_feature_cache(project_info)
    metrics_payload = collect_final_metrics(project_info)
    model_card_path = build_model_card(feature_summary, metrics_payload)
    log_final_artifacts_to_mlflow(project_info, feature_summary, metrics_payload, model_card_path)


ftd_eeg_mlflow_final_pipeline()
