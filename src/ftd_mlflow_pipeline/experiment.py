from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from pyriemann.classification import FgMDM, class_distinctiveness
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .connectivity import BANDS, METRICS, FeatureSet, compute_feature_set
from .precomputed_features import load_precomputed_feature_catalog


PROBLEMS = {
    "ad_hc": ("A", "C", "AD", "HC"),
    "ftd_hc": ("F", "C", "FTD", "HC"),
    "ftd_ad": ("F", "A", "FTD", "AD"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    cache_dir: Path
    tracking_uri: str
    experiment_name: str
    feature_source: str = "from_epochs"
    precomputed_dir: Path | None = None
    bands: tuple[str, ...] = BANDS
    metrics: tuple[str, ...] = METRICS
    fgmdm_metric: str = "logeuclid"
    filter_ratio: float = 0.5
    inner_folds: int = 5
    outer_limit: int | None = None
    random_state: int = 42
    elasticnet_alpha: float = 1.0
    elasticnet_l1_ratio: float = 0.15


@dataclass
class BaseModelPrediction:
    name: str
    oof_subject_probs: np.ndarray
    test_subject_prob: float


def _binary_subset(feature_set: FeatureSet, positive_code: str, negative_code: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isin(feature_set.group_codes, [positive_code, negative_code])
    matrices = feature_set.matrices[mask]
    subjects = feature_set.subject_ids[mask]
    labels = (feature_set.group_codes[mask] == positive_code).astype(int)
    return matrices, labels, subjects


def _subject_frame(labels: np.ndarray, subject_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"subject_id": subject_ids, "label": labels})
    subject_frame = frame.drop_duplicates("subject_id").sort_values("subject_id").reset_index(drop=True)
    counts = frame.groupby("subject_id").size().rename("epoch_count")
    subject_frame = subject_frame.merge(counts, on="subject_id", how="left")
    subject_frame["sample_weight"] = 1.0 / subject_frame["epoch_count"]
    return subject_frame


def _epoch_weights(subject_ids: np.ndarray) -> np.ndarray:
    counts = pd.Series(subject_ids).value_counts()
    return np.asarray([1.0 / counts[sid] for sid in subject_ids], dtype=np.float64)


def _safe_inner_folds(labels: np.ndarray, requested_folds: int) -> int:
    class_counts = np.bincount(labels.astype(int))
    min_count = int(class_counts.min())
    if min_count < 2:
        raise ValueError("At least two subjects per class are required for inner cross-validation.")
    return int(min(requested_folds, min_count))


def _aggregate_probs_by_subject(probabilities: np.ndarray, subject_ids: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"subject_id": subject_ids, "probability": probabilities})
    return frame.groupby("subject_id")["probability"].mean()


def _fit_base_oof_predictions(
    matrices: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    train_subject_ids: set[str],
    test_subject_id: str,
    fgmdm_metric: str,
    inner_folds: int,
    random_state: int,
) -> BaseModelPrediction:
    train_mask = np.isin(subject_ids, list(train_subject_ids))
    test_mask = subject_ids == test_subject_id

    X_train = matrices[train_mask]
    y_train = labels[train_mask]
    groups_train = subject_ids[train_mask]
    X_test = matrices[test_mask]
    groups_test = subject_ids[test_mask]

    train_subject_frame = _subject_frame(y_train, groups_train)
    subject_order = train_subject_frame["subject_id"].to_numpy()
    inner_splits = _safe_inner_folds(train_subject_frame["label"].to_numpy(), inner_folds)
    inner_cv = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    epoch_weights = _epoch_weights(groups_train)

    oof_probs = np.empty(y_train.shape[0], dtype=np.float64)
    for inner_train_idx, inner_valid_idx in inner_cv.split(X_train, y_train, groups_train):
        model = FgMDM(metric=fgmdm_metric)
        model.fit(
            X_train[inner_train_idx],
            y_train[inner_train_idx],
            sample_weight=epoch_weights[inner_train_idx],
        )
        oof_probs[inner_valid_idx] = model.predict_proba(X_train[inner_valid_idx])[:, 1]

    aggregated_train = _aggregate_probs_by_subject(oof_probs, groups_train)
    oof_subject_probs = aggregated_train.reindex(subject_order).to_numpy()

    final_model = FgMDM(metric=fgmdm_metric)
    final_model.fit(X_train, y_train, sample_weight=epoch_weights)
    test_probs = final_model.predict_proba(X_test)[:, 1]
    aggregated_test = _aggregate_probs_by_subject(test_probs, groups_test)
    return BaseModelPrediction(
        name="",
        oof_subject_probs=oof_subject_probs,
        test_subject_prob=float(aggregated_test.loc[test_subject_id]),
    )


def _meta_score(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    selected_columns: list[int],
    inner_folds: int,
    random_state: int,
    alpha: float,
    l1_ratio: float,
) -> float:
    n_splits = _safe_inner_folds(y, inner_folds)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, valid_idx in splitter.split(X[:, selected_columns], y):
        model = _fit_meta_model(
            _build_meta_model(alpha, l1_ratio, random_state),
            X[train_idx][:, selected_columns],
            y[train_idx],
            sample_weight[train_idx],
        )
        valid_probs = model.predict_proba(X[valid_idx][:, selected_columns])[:, 1]
        scores.append(roc_auc_score(y[valid_idx], valid_probs))
    return float(np.mean(scores))


def _build_meta_model(alpha: float, l1_ratio: float, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        C=1.0 / alpha,
        max_iter=3000,
        random_state=random_state,
    )


def _fit_meta_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*'penalty' was deprecated.*",
            category=FutureWarning,
        )
        model.fit(X, y, sample_weight=sample_weight)
    return model


def _greedy_wrapper_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sample_weight: np.ndarray,
    inner_folds: int,
    random_state: int,
    alpha: float,
    l1_ratio: float,
) -> list[int]:
    selected: list[int] = []
    remaining = list(range(X.shape[1]))
    best_score = -math.inf
    while remaining:
        trial_scores: list[tuple[float, int]] = []
        for candidate in remaining:
            score = _meta_score(
                X,
                y,
                sample_weight,
                selected + [candidate],
                inner_folds,
                random_state,
                alpha,
                l1_ratio,
            )
            trial_scores.append((score, candidate))
        trial_scores.sort(reverse=True)
        candidate_score, candidate_idx = trial_scores[0]
        if not selected or candidate_score > best_score:
            selected.append(candidate_idx)
            remaining.remove(candidate_idx)
            best_score = candidate_score
        else:
            break
    return selected


def _evaluate_subject_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn) if (tp + fn) else 0.0),
        "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
    }


def _limit_subject_frame_stratified(
    subject_frame: pd.DataFrame,
    outer_limit: int,
    random_state: int,
) -> pd.DataFrame:
    if outer_limit >= len(subject_frame):
        return subject_frame.reset_index(drop=True)

    if outer_limit < subject_frame["label"].nunique() * 2:
        raise ValueError("outer_limit is too small to keep both classes in train/test folds.")

    rng = np.random.default_rng(random_state)
    class_sizes = subject_frame["label"].value_counts().sort_index()
    quotas = {
        label: max(1, int(round(outer_limit * count / len(subject_frame))))
        for label, count in class_sizes.items()
    }

    while sum(quotas.values()) > outer_limit:
        label = max(quotas, key=quotas.get)
        if quotas[label] > 1:
            quotas[label] -= 1
    while sum(quotas.values()) < outer_limit:
        label = max(class_sizes.index, key=lambda lbl: class_sizes[lbl] - quotas[lbl])
        quotas[label] += 1

    samples = []
    for label, quota in quotas.items():
        subset = subject_frame[subject_frame["label"] == label]
        chosen = subset.sample(n=quota, random_state=int(rng.integers(0, 1_000_000)))
        samples.append(chosen)
    limited = pd.concat(samples).sort_values("subject_id").reset_index(drop=True)
    return limited


def _load_feature_catalog(config: ExperimentConfig) -> dict[str, FeatureSet]:
    if config.feature_source == "from_precomputed":
        if config.precomputed_dir is None:
            raise ValueError("precomputed_dir is required when feature_source='from_precomputed'.")
        return load_precomputed_feature_catalog(
            config.precomputed_dir,
            config.bands,
            config.metrics,
        )

    catalog: dict[str, FeatureSet] = {}
    for band in config.bands:
        for metric in config.metrics:
            key = f"{band}__{metric}"
            catalog[key] = compute_feature_set(config.data_dir, config.cache_dir, band, metric)
    return catalog


def run_problem(config: ExperimentConfig, problem_name: str) -> dict[str, object]:
    positive_code, negative_code, positive_name, negative_name = PROBLEMS[problem_name]
    feature_catalog = _load_feature_catalog(config)
    first_feature = next(iter(feature_catalog.values()))
    _, labels_all, subjects_all = _binary_subset(first_feature, positive_code, negative_code)
    subject_frame = _subject_frame(labels_all, subjects_all)
    if config.outer_limit is not None:
        subject_frame = _limit_subject_frame_stratified(
            subject_frame,
            config.outer_limit,
            config.random_state,
        )

    outer_predictions: list[dict[str, object]] = []
    fold_artifacts: list[dict[str, object]] = []

    for fold_idx, test_row in subject_frame.iterrows():
        test_subject_id = str(test_row["subject_id"])
        train_subject_frame = subject_frame[subject_frame["subject_id"] != test_subject_id].reset_index(drop=True)
        train_subject_ids = set(train_subject_frame["subject_id"])

        distinctiveness_scores: dict[str, float] = {}
        for name, feature_set in feature_catalog.items():
            matrices, labels, subject_ids = _binary_subset(feature_set, positive_code, negative_code)
            train_mask = np.isin(subject_ids, list(train_subject_ids))
            distinctiveness_scores[name] = float(
                class_distinctiveness(
                    matrices[train_mask],
                    labels[train_mask],
                    metric=config.fgmdm_metric,
                )
            )

        ranked = sorted(distinctiveness_scores.items(), key=lambda item: item[1], reverse=True)
        retain_count = max(1, int(math.ceil(len(ranked) * config.filter_ratio)))
        retained_names = [name for name, _ in ranked[:retain_count]]

        meta_columns: list[np.ndarray] = []
        test_feature_values: list[float] = []
        for name in retained_names:
            feature_set = feature_catalog[name]
            matrices, labels, subject_ids = _binary_subset(feature_set, positive_code, negative_code)
            prediction = _fit_base_oof_predictions(
                matrices=matrices,
                labels=labels,
                subject_ids=subject_ids,
                train_subject_ids=train_subject_ids,
                test_subject_id=test_subject_id,
                fgmdm_metric=config.fgmdm_metric,
                inner_folds=config.inner_folds,
                random_state=config.random_state,
            )
            meta_columns.append(prediction.oof_subject_probs)
            test_feature_values.append(prediction.test_subject_prob)

        X_meta_train = np.column_stack(meta_columns)
        y_meta_train = train_subject_frame["label"].to_numpy()
        sample_weight = train_subject_frame["sample_weight"].to_numpy(dtype=np.float64)
        selected_indices = _greedy_wrapper_selection(
            X=X_meta_train,
            y=y_meta_train,
            feature_names=retained_names,
            sample_weight=sample_weight,
            inner_folds=config.inner_folds,
            random_state=config.random_state,
            alpha=config.elasticnet_alpha,
            l1_ratio=config.elasticnet_l1_ratio,
        )

        if not selected_indices:
            selected_indices = [0]

        meta_model = _fit_meta_model(
            _build_meta_model(
                config.elasticnet_alpha,
                config.elasticnet_l1_ratio,
                config.random_state,
            ),
            X_meta_train[:, selected_indices],
            y_meta_train,
            sample_weight,
        )
        X_meta_test = np.asarray(test_feature_values, dtype=np.float64)[selected_indices].reshape(1, -1)
        test_prob = float(meta_model.predict_proba(X_meta_test)[0, 1])
        outer_predictions.append(
            {
                "subject_id": test_subject_id,
                "y_true": int(test_row["label"]),
                "y_prob": test_prob,
                "selected_features": [retained_names[idx] for idx in selected_indices],
            }
        )
        fold_artifacts.append(
            {
                "fold_index": fold_idx,
                "test_subject_id": test_subject_id,
                "retained_features": retained_names,
                "selected_features": [retained_names[idx] for idx in selected_indices],
                "top_distinctiveness": ranked[:10],
            }
        )

    predictions_df = pd.DataFrame(outer_predictions)
    metrics = _evaluate_subject_metrics(
        predictions_df["y_true"].to_numpy(),
        predictions_df["y_prob"].to_numpy(),
    )

    return {
        "problem_name": problem_name,
        "positive_class": positive_name,
        "negative_class": negative_name,
        "metrics": metrics,
        "predictions": predictions_df,
        "fold_artifacts": fold_artifacts,
    }


def run_experiment(config: ExperimentConfig, problem_names: list[str]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(config.cache_dir.parent / "mplconfig"))
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name="paper_pipeline") as parent_run:
        mlflow.log_params(
            {
                "bands": ",".join(config.bands),
                "metrics": ",".join(config.metrics),
                "feature_source": config.feature_source,
                "fgmdm_metric": config.fgmdm_metric,
                "filter_ratio": config.filter_ratio,
                "inner_folds": config.inner_folds,
                "outer_limit": config.outer_limit if config.outer_limit is not None else "all",
                "elasticnet_alpha": config.elasticnet_alpha,
                "elasticnet_l1_ratio": config.elasticnet_l1_ratio,
            }
        )
        if config.precomputed_dir is not None:
            mlflow.log_param("precomputed_dir", str(config.precomputed_dir))
        mlflow.log_text(
            "Paper-driven pipeline using functional connectivity matrices, FgMDM base classifiers, subject-level LOSO-CV, and MLflow tracking.",
            "paper_method_summary.txt",
        )

        for problem_name in problem_names:
            results = run_problem(config, problem_name)
            with mlflow.start_run(run_name=problem_name, nested=True):
                mlflow.log_params(
                    {
                        "problem_name": problem_name,
                        "positive_class": results["positive_class"],
                        "negative_class": results["negative_class"],
                    }
                )
                mlflow.log_metrics(results["metrics"])
                with TemporaryDirectory() as tmp_dir_name:
                    tmp_dir = Path(tmp_dir_name)
                    predictions_path = tmp_dir / f"{problem_name}_predictions.csv"
                    folds_path = tmp_dir / f"{problem_name}_folds.json"
                    results["predictions"].to_csv(predictions_path, index=False)
                    folds_path.write_text(json.dumps(results["fold_artifacts"], indent=2))
                    mlflow.log_artifact(predictions_path)
                    mlflow.log_artifact(folds_path)
