from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mplconfig"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ftd_mlflow_pipeline.connectivity import BANDS, METRICS
from ftd_mlflow_pipeline.experiment import ExperimentConfig, PROBLEMS, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLflow experiments for AD/FTD/HC EEG classification.")
    parser.add_argument(
        "--feature-source",
        choices=["from_epochs", "from_precomputed"],
        default="from_epochs",
        help="Choose whether to compute connectivity from Cleaned_Epochs or load precomputed domain features.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "Cleaned_Epochs",
        help="Directory containing *_band-epo.fif files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache" / "connectivity",
        help="Directory used to cache computed connectivity matrices.",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=Path,
        default=ROOT / "Final_MultiDomain_Features_Role3(1)",
        help="Directory containing precomputed feature .npy files for the fast branch.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=f"file:{ROOT / 'mlruns'}",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default="ftd_ad_hc_paper_pipeline",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--problem",
        choices=["all", *PROBLEMS.keys()],
        default="all",
        help="Binary problem to run. 'all' runs ad_hc, ftd_hc, and ftd_ad.",
    )
    parser.add_argument(
        "--bands",
        default=",".join(BANDS),
        help="Comma-separated list of bands.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(METRICS),
        help="Comma-separated list of connectivity metrics.",
    )
    parser.add_argument(
        "--fgmdm-metric",
        choices=["riemann", "logeuclid", "euclid"],
        default="logeuclid",
        help="Distance metric used by FgMDM.",
    )
    parser.add_argument(
        "--filter-ratio",
        type=float,
        default=0.5,
        help="Fraction of base feature sets retained after class distinctiveness ranking.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=5,
        help="Number of inner CV folds used for stacking.",
    )
    parser.add_argument(
        "--outer-limit",
        type=int,
        default=None,
        help="Optional limit on the number of outer LOSO folds for smoke tests.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducible splits.",
    )
    parser.add_argument(
        "--elasticnet-alpha",
        type=float,
        default=1.0,
        help="Regularization strength parameter reported in the paper.",
    )
    parser.add_argument(
        "--elasticnet-l1-ratio",
        type=float,
        default=0.15,
        help="Elastic Net l1_ratio reported in the paper.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_bands = tuple(band.strip() for band in args.bands.split(",") if band.strip())
    selected_metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())

    invalid_bands = sorted(set(selected_bands) - set(BANDS))
    invalid_metrics = sorted(set(selected_metrics) - set(METRICS))
    if invalid_bands:
        raise SystemExit(f"Unsupported bands: {', '.join(invalid_bands)}")
    if invalid_metrics:
        raise SystemExit(f"Unsupported metrics: {', '.join(invalid_metrics)}")

    config = ExperimentConfig(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        feature_source=args.feature_source,
        precomputed_dir=args.precomputed_dir,
        bands=selected_bands,
        metrics=selected_metrics,
        fgmdm_metric=args.fgmdm_metric,
        filter_ratio=args.filter_ratio,
        inner_folds=args.inner_folds,
        outer_limit=args.outer_limit,
        random_state=args.random_state,
        elasticnet_alpha=args.elasticnet_alpha,
        elasticnet_l1_ratio=args.elasticnet_l1_ratio,
    )
    problem_names = list(PROBLEMS) if args.problem == "all" else [args.problem]
    run_experiment(config, problem_names)


if __name__ == "__main__":
    main()
