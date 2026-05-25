from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task


PROJECT_ROOT = Path(os.environ.get("FTD_EEG_PROJECT_ROOT", "/opt/airflow/project"))
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
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_LOSOCV_EXPERIMENT_NAME", "ftd_ad_hc_losocv_airflow")
OUTPUT_DIR = PROJECT_ROOT / "notebook_outputs" / "airflow_losocv_pipeline"

BANDS = os.environ.get("FTD_EEG_LOSOCV_BANDS", "delta,theta,alpha,beta,gamma")
METRICS = os.environ.get(
    "FTD_EEG_LOSOCV_METRICS",
    "cov,corr,xcov,xcorr,csd,coh,mi,ecc,aecov,aecorr,plv,wplv",
)
INNER_FOLDS = os.environ.get("FTD_EEG_LOSOCV_INNER_FOLDS", "5")
OUTER_LIMIT = os.environ.get("FTD_EEG_LOSOCV_OUTER_LIMIT", "").strip()
FGMDM_METRIC = os.environ.get("FTD_EEG_FGMDM_METRIC", "logeuclid")
FILTER_RATIO = os.environ.get("FTD_EEG_FILTER_RATIO", "0.5")
ELASTICNET_ALPHA = os.environ.get("FTD_EEG_ELASTICNET_ALPHA", "1.0")
ELASTICNET_L1_RATIO = os.environ.get("FTD_EEG_ELASTICNET_L1_RATIO", "0.15")

PROBLEMS = ("ad_hc", "ftd_hc", "ftd_ad")


@dag(
    dag_id="ftd_eeg_losocv_mlflow_pipeline",
    description="Ensure connectivity cache, run LOSOCV stacked ensemble, and log results to MLflow.",
    start_date=datetime(2026, 5, 25),
    schedule=None,
    catchup=False,
    tags=["eeg", "ftd", "ad", "losocv", "mlflow", "training"],
)
def ftd_eeg_losocv_mlflow_pipeline():
    @task
    def ensure_feature_cache() -> dict[str, object]:
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
        return json.loads(completed.stdout)

    @task
    def train_losocv_problem(problem_name: str, cache_status: dict[str, object]) -> dict[str, object]:
        start = time.perf_counter()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        script_path = PROJECT_ROOT / "scripts" / "run_mlflow_experiment.py"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing MLflow training script: {script_path}")

        command = [
            sys.executable,
            str(script_path),
            "--feature-source",
            "from_precomputed",
            "--precomputed-dir",
            str(FEATURE_DIR),
            "--tracking-uri",
            MLFLOW_TRACKING_URI,
            "--experiment-name",
            MLFLOW_EXPERIMENT_NAME,
            "--problem",
            problem_name,
            "--bands",
            BANDS,
            "--metrics",
            METRICS,
            "--fgmdm-metric",
            FGMDM_METRIC,
            "--filter-ratio",
            FILTER_RATIO,
            "--inner-folds",
            INNER_FOLDS,
            "--elasticnet-alpha",
            ELASTICNET_ALPHA,
            "--elasticnet-l1-ratio",
            ELASTICNET_L1_RATIO,
        ]
        if OUTER_LIMIT:
            command.extend(["--outer-limit", OUTER_LIMIT])

        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
            text=True,
            capture_output=True,
        )

        log_path = OUTPUT_DIR / f"{problem_name}_losocv_airflow.log"
        log_path.write_text(
            "\n".join(
                [
                    f"problem={problem_name}",
                    f"feature_dir={FEATURE_DIR}",
                    f"cache_action={cache_status.get('action', cache_status.get('feature_cache_action', 'unknown'))}",
                    f"mlflow_tracking_uri={MLFLOW_TRACKING_URI}",
                    f"mlflow_experiment_name={MLFLOW_EXPERIMENT_NAME}",
                    "command=" + " ".join(command),
                    "",
                    "STDOUT:",
                    completed.stdout,
                    "",
                    "STDERR:",
                    completed.stderr,
                ]
            ),
            encoding="utf-8",
        )

        return {
            "problem": problem_name,
            "seconds": time.perf_counter() - start,
            "log_path": str(log_path),
            "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        }

    @task
    def summarize_losocv_runs(run_results: list[dict[str, object]]) -> str:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary = {
            "feature_dir": str(FEATURE_DIR),
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "mlflow_experiment_name": MLFLOW_EXPERIMENT_NAME,
            "problems": run_results,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary_path = OUTPUT_DIR / "losocv_airflow_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(summary_path)

    cache_status = ensure_feature_cache()
    run_results = [
        train_losocv_problem.override(task_id=f"train_{problem_name}_losocv")(problem_name, cache_status)
        for problem_name in PROBLEMS
    ]
    summarize_losocv_runs(run_results)


ftd_eeg_losocv_mlflow_pipeline()
