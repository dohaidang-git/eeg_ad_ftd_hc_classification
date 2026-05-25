from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ftd_mlflow_pipeline.connectivity import BANDS, METRICS, compute_feature_set


EXPECTED_FEATURE_NAMES = tuple(f"{band}_{metric}" for band in BANDS for metric in METRICS)


def normalize_subject_ids(subject_ids: np.ndarray) -> np.ndarray:
    normalized: list[str] = []
    for subject_id in subject_ids.astype(str):
        if subject_id.startswith("sub-"):
            normalized.append(subject_id)
        else:
            normalized.append(f"sub-{subject_id.zfill(3)}")
    return np.asarray(normalized)


def missing_cache_files(output_dir: Path) -> list[Path]:
    required = [
        output_dir / "labels.npy",
        output_dir / "subject_ids.npy",
        output_dir / "feature_metadata.csv",
        output_dir / "all_feature_names.npy",
    ]
    required.extend(output_dir / f"{feature_name}.npy" for feature_name in EXPECTED_FEATURE_NAMES)
    return [path for path in required if not path.exists()]


def is_cache_complete(output_dir: Path) -> bool:
    return output_dir.exists() and not missing_cache_files(output_dir)


def write_manifest(output_dir: Path, action: str, seconds: float) -> None:
    manifest = {
        "action": action,
        "seconds": seconds,
        "feature_count": len(EXPECTED_FEATURE_NAMES),
        "bands": list(BANDS),
        "metrics": list(METRICS),
        "output_dir": str(output_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_connectivity_cache(
    input_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    check_only: bool = False,
) -> dict[str, object]:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if is_cache_complete(output_dir):
        seconds = time.perf_counter() - start
        if check_only:
            return {
                "action": "cache_complete",
                "output_dir": str(output_dir),
                "missing_files": [],
                "seconds": seconds,
            }
        write_manifest(output_dir, "reused_existing_cache", seconds)
        return {
            "action": "reused_existing_cache",
            "output_dir": str(output_dir),
            "missing_files": [],
            "seconds": seconds,
        }

    missing_before = missing_cache_files(output_dir)
    if check_only:
        return {
            "action": "cache_incomplete",
            "output_dir": str(output_dir),
            "missing_files": [str(path) for path in missing_before],
            "seconds": time.perf_counter() - start,
        }

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Connectivity cache is incomplete and input epoch directory is missing: {input_dir}"
        )

    labels_path = output_dir / "labels.npy"
    subject_ids_path = output_dir / "subject_ids.npy"
    labels_reference = np.load(labels_path, allow_pickle=True).astype(str) if labels_path.exists() else None
    subjects_reference = (
        normalize_subject_ids(np.load(subject_ids_path, allow_pickle=True))
        if subject_ids_path.exists()
        else None
    )

    metadata_rows: list[dict[str, object]] = []
    computed_features: list[str] = []
    reused_features: list[str] = []

    for band in BANDS:
        for metric in METRICS:
            feature_name = f"{band}_{metric}"
            output_path = output_dir / f"{feature_name}.npy"

            if output_path.exists() and labels_reference is not None and subjects_reference is not None:
                matrices = np.load(output_path, mmap_mode="r", allow_pickle=False)
                reused_features.append(feature_name)
            else:
                feature_set = compute_feature_set(input_dir, cache_dir, band, metric)
                current_labels = feature_set.group_codes.astype(str)
                current_subjects = normalize_subject_ids(feature_set.subject_ids)

                if labels_reference is None:
                    labels_reference = current_labels
                    np.save(labels_path, labels_reference)
                elif not np.array_equal(labels_reference.astype(str), current_labels.astype(str)):
                    raise ValueError(f"Label alignment mismatch while computing {feature_name}")

                if subjects_reference is None:
                    subjects_reference = current_subjects
                    np.save(subject_ids_path, subjects_reference)
                elif not np.array_equal(subjects_reference.astype(str), current_subjects.astype(str)):
                    raise ValueError(f"Subject alignment mismatch while computing {feature_name}")

                np.save(output_path, feature_set.matrices.astype(np.float64))
                matrices = feature_set.matrices
                computed_features.append(feature_name)

            metadata_rows.append(
                {
                    "feature": feature_name,
                    "band": band,
                    "metric": metric,
                    "path": str(output_path),
                    "n_epochs": int(matrices.shape[0]),
                    "n_channels": int(matrices.shape[1]),
                    "source": "computed" if feature_name in computed_features else "reused",
                }
            )

    if labels_reference is None or subjects_reference is None:
        raise RuntimeError("Could not infer labels/subject_ids for feature cache.")

    np.save(output_dir / "labels.npy", labels_reference.astype(str))
    np.save(output_dir / "subject_ids.npy", subjects_reference.astype(str))
    np.save(output_dir / "all_feature_names.npy", np.asarray(EXPECTED_FEATURE_NAMES))
    np.save(output_dir / "top_features_name.npy", np.asarray(EXPECTED_FEATURE_NAMES))
    pd.DataFrame(metadata_rows).to_csv(output_dir / "feature_metadata.csv", index=False)

    missing_after = missing_cache_files(output_dir)
    if missing_after:
        raise RuntimeError("Feature cache is still incomplete:\n" + "\n".join(map(str, missing_after)))

    seconds = time.perf_counter() - start
    action = "computed_missing_connectivity" if computed_features else "repaired_cache_metadata"
    write_manifest(output_dir, action, seconds)
    return {
        "action": action,
        "output_dir": str(output_dir),
        "computed_features": computed_features,
        "reused_features": reused_features,
        "feature_count": len(EXPECTED_FEATURE_NAMES),
        "seconds": seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure a complete 60-feature EEG connectivity cache exists.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "Cleaned_Epochs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Full_MultiDomain_Features_Role3_v5")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "connectivity")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = ensure_connectivity_cache(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        check_only=args.check_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mplconfig"))
    main()
