from __future__ import annotations

from pathlib import Path

import numpy as np

from .connectivity import FeatureSet


FEATURE_NAME_CANDIDATES = (
    "top_features_name.npy",
    "all_feature_names.npy",
    "computed_feature_names.npy",
)


def _normalize_subject_ids(subject_ids: np.ndarray) -> np.ndarray:
    normalized: list[str] = []
    for subject_id in subject_ids.astype(str):
        if subject_id.startswith("sub-"):
            normalized.append(subject_id)
        else:
            normalized.append(f"sub-{subject_id.zfill(3)}")
    return np.asarray(normalized)


def _load_feature_names(precomputed_dir: Path) -> list[str]:
    for file_name in FEATURE_NAME_CANDIDATES:
        path = precomputed_dir / file_name
        if path.exists():
            return np.load(path, allow_pickle=True).astype(str).tolist()

    metadata_path = precomputed_dir / "feature_metadata.csv"
    if metadata_path.exists():
        import pandas as pd

        metadata = pd.read_csv(metadata_path)
        if "feature" in metadata.columns:
            return metadata["feature"].astype(str).tolist()

    feature_names = sorted(
        path.stem
        for path in precomputed_dir.glob("*.npy")
        if path.stem not in {"labels", "subject_ids", "all_feature_names", "top_features_name"}
    )
    if feature_names:
        return feature_names

    raise FileNotFoundError(
        "Could not find feature-name metadata in precomputed directory. "
        f"Checked: {', '.join(FEATURE_NAME_CANDIDATES)}, feature_metadata.csv."
    )


def load_precomputed_feature_catalog(
    precomputed_dir: Path,
    selected_bands: tuple[str, ...],
    selected_metrics: tuple[str, ...],
) -> dict[str, FeatureSet]:
    if not precomputed_dir.exists():
        raise FileNotFoundError(f"Precomputed feature directory not found: {precomputed_dir}")

    feature_names = _load_feature_names(precomputed_dir)
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
