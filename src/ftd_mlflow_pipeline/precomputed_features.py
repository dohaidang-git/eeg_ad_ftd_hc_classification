from __future__ import annotations

from pathlib import Path

import numpy as np

from .connectivity import FeatureSet


def _normalize_subject_ids(subject_ids: np.ndarray) -> np.ndarray:
    normalized: list[str] = []
    for subject_id in subject_ids.astype(str):
        if subject_id.startswith("sub-"):
            normalized.append(subject_id)
        else:
            normalized.append(f"sub-{subject_id.zfill(3)}")
    return np.asarray(normalized)


def load_precomputed_feature_catalog(
    precomputed_dir: Path,
    selected_bands: tuple[str, ...],
    selected_metrics: tuple[str, ...],
) -> dict[str, FeatureSet]:
    if not precomputed_dir.exists():
        raise FileNotFoundError(f"Precomputed feature directory not found: {precomputed_dir}")

    feature_names = np.load(precomputed_dir / "top_features_name.npy", allow_pickle=True).tolist()
    labels = np.load(precomputed_dir / "labels.npy", allow_pickle=True).astype(str)
    subject_ids = _normalize_subject_ids(np.load(precomputed_dir / "subject_ids.npy", allow_pickle=True))

    catalog: dict[str, FeatureSet] = {}
    for feature_name in feature_names:
        band, metric = feature_name.split("_", maxsplit=1)
        if band not in selected_bands or metric not in selected_metrics:
            continue
        matrices = np.load(precomputed_dir / f"{feature_name}.npy", allow_pickle=True)
        key = f"{band}__{metric}"
        catalog[key] = FeatureSet(
            band=band,
            metric=metric,
            matrices=matrices.astype(np.float64),
            group_codes=labels,
            subject_ids=subject_ids,
        )

    if not catalog:
        raise ValueError(
            "No precomputed features matched the requested bands/metrics. "
            f"Requested bands={selected_bands}, metrics={selected_metrics}."
        )
    return catalog
