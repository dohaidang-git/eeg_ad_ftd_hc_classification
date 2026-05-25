from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mne
import numpy as np
from scipy.signal import coherence, correlate, csd, hilbert


BANDS = ("delta", "theta", "alpha", "beta", "gamma")
BAND_RANGES = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "gamma": (30.0, 45.0),
}
CACHE_VERSION = "v4"
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


@dataclass(frozen=True)
class FeatureSet:
    band: str
    metric: str
    matrices: np.ndarray
    group_codes: np.ndarray
    subject_ids: np.ndarray


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(_symmetrize(matrix))
    except np.linalg.LinAlgError:
        return False
    return True


def nearest_spd(matrix: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix to the nearest SPD matrix using Higham's method."""
    sym = _symmetrize(np.asarray(matrix, dtype=np.float64))
    _, singular_values, vh = np.linalg.svd(sym)
    polar_factor = vh.T @ np.diag(singular_values) @ vh
    candidate = _symmetrize((sym + polar_factor) / 2.0)

    if not _is_positive_definite(candidate):
        identity = np.eye(candidate.shape[0], dtype=np.float64)
        spacing = np.spacing(max(np.linalg.norm(sym), 1.0))
        k = 1
        while not _is_positive_definite(candidate):
            min_eigenvalue = float(np.min(np.linalg.eigvalsh(candidate)))
            candidate += identity * (-min_eigenvalue * k**2 + spacing)
            candidate = _symmetrize(candidate)
            k += 1

    min_eigenvalue = float(np.min(np.linalg.eigvalsh(candidate)))
    if min_eigenvalue < min_eig:
        candidate += np.eye(candidate.shape[0], dtype=np.float64) * (min_eig - min_eigenvalue)
    return _symmetrize(candidate)


def stabilize_spd(matrix: np.ndarray) -> np.ndarray:
    matrix = nearest_spd(matrix)
    min_eig = float(np.min(np.linalg.eigvalsh(matrix)))
    if min_eig <= 0:
        matrix = matrix + np.eye(matrix.shape[0], dtype=np.float64) * (abs(min_eig) + 1e-8)
    return _symmetrize(matrix)


def freedman_diaconis_bins(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=np.float64).ravel()
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return 16
    bin_width = 2.0 * iqr / np.cbrt(values.size)
    if bin_width <= 0:
        return 16
    bins = int(np.ceil((values.max() - values.min()) / bin_width))
    return int(np.clip(bins, 8, 128))


def _safe_corrcoef(values: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(values)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _cross_covariance_matrix(epoch: np.ndarray) -> np.ndarray:
    centered = epoch - epoch.mean(axis=1, keepdims=True)
    n_channels, n_times = centered.shape
    output = np.empty((n_channels, n_channels), dtype=np.float64)
    variances = centered.var(axis=1)
    for i in range(n_channels):
        output[i, i] = variances[i]
        for j in range(i + 1, n_channels):
            corr_seq = correlate(centered[i], centered[j], mode="full", method="fft") / n_times
            value = corr_seq[np.argmax(np.abs(corr_seq))]
            output[i, j] = value
            output[j, i] = value
    return output


def _cross_correlation_matrix(epoch: np.ndarray) -> np.ndarray:
    xcov = _cross_covariance_matrix(epoch)
    std = epoch.std(axis=1, ddof=0)
    denom = np.outer(std, std)
    matrix = np.divide(xcov, denom, out=np.zeros_like(xcov), where=denom > 0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _band_frequency_mask(freqs: np.ndarray, band: str | None) -> np.ndarray:
    if band is None:
        return np.ones(freqs.shape, dtype=bool)
    low, high = BAND_RANGES[band]
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return np.ones(freqs.shape, dtype=bool)
    return mask


def _csd_matrix(epoch: np.ndarray, sfreq: float, band: str | None) -> np.ndarray:
    n_channels = epoch.shape[0]
    nperseg = min(256, epoch.shape[1])
    matrix = np.empty((n_channels, n_channels), dtype=np.float64)
    for i in range(n_channels):
        freqs, pxx = csd(epoch[i], epoch[i], fs=sfreq, nperseg=nperseg)
        band_mask = _band_frequency_mask(freqs, band)
        matrix[i, i] = float(np.mean(np.abs(pxx[band_mask])))
        for j in range(i + 1, n_channels):
            _, pxy = csd(epoch[i], epoch[j], fs=sfreq, nperseg=nperseg)
            value = float(np.mean(np.abs(pxy[band_mask])))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _coherence_matrix(epoch: np.ndarray, sfreq: float, band: str | None) -> np.ndarray:
    n_channels = epoch.shape[0]
    nperseg = min(256, epoch.shape[1])
    matrix = np.eye(n_channels, dtype=np.float64)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh_values = coherence(epoch[i], epoch[j], fs=sfreq, nperseg=nperseg)
            band_mask = _band_frequency_mask(freqs, band)
            value = float(np.mean(coh_values[band_mask]))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _entropy_from_hist(hist: np.ndarray) -> float:
    probs = hist.astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def _mutual_information_matrix(epoch: np.ndarray, n_bins: int) -> np.ndarray:
    n_channels = epoch.shape[0]
    matrix = np.empty((n_channels, n_channels), dtype=np.float64)
    entropies: list[float] = []
    for i in range(n_channels):
        hist, _ = np.histogram(epoch[i], bins=n_bins)
        entropies.append(_entropy_from_hist(hist))
    entropies_arr = np.asarray(entropies, dtype=np.float64)

    for i in range(n_channels):
        matrix[i, i] = entropies_arr[i]
        for j in range(i + 1, n_channels):
            hist2d, _, _ = np.histogram2d(epoch[i], epoch[j], bins=n_bins)
            hxy = _entropy_from_hist(hist2d)
            value = entropies_arr[i] + entropies_arr[j] - hxy
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _entropy_correlation_matrix(epoch: np.ndarray, n_bins: int) -> np.ndarray:
    mi = _mutual_information_matrix(epoch, n_bins)
    entropy = np.diag(mi).copy()
    denom = np.sqrt(np.outer(entropy, entropy))
    matrix = np.divide(mi, denom, out=np.zeros_like(mi), where=denom > 0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _analytic_signal_matrices(epoch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    analytic = hilbert(epoch, axis=1)
    amplitude = np.abs(analytic)
    return analytic, amplitude


def _plv_matrix(epoch: np.ndarray) -> np.ndarray:
    analytic, _ = _analytic_signal_matrices(epoch)
    unit = np.divide(analytic, np.abs(analytic), out=np.zeros_like(analytic), where=np.abs(analytic) > 0)
    matrix = np.abs(unit @ unit.conj().T) / unit.shape[1]
    np.fill_diagonal(matrix, 1.0)
    return matrix.real


def _wplv_matrix(epoch: np.ndarray) -> np.ndarray:
    analytic, amplitude = _analytic_signal_matrices(epoch)
    numerator = np.abs(analytic @ analytic.conj().T) / analytic.shape[1]
    denominator = (amplitude @ amplitude.T) / amplitude.shape[1]
    matrix = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    np.fill_diagonal(matrix, 1.0)
    return matrix.real


def compute_connectivity_matrix(
    epoch: np.ndarray,
    metric: str,
    sfreq: float,
    n_bins: int,
    band: str | None = None,
) -> np.ndarray:
    if metric == "cov":
        matrix = np.cov(epoch, bias=True)
    elif metric == "corr":
        matrix = _safe_corrcoef(epoch)
    elif metric == "xcov":
        matrix = _cross_covariance_matrix(epoch)
    elif metric == "xcorr":
        matrix = _cross_correlation_matrix(epoch)
    elif metric == "csd":
        matrix = _csd_matrix(epoch, sfreq, band)
    elif metric == "coh":
        matrix = _coherence_matrix(epoch, sfreq, band)
    elif metric == "mi":
        matrix = _mutual_information_matrix(epoch, n_bins)
    elif metric == "ecc":
        matrix = _entropy_correlation_matrix(epoch, n_bins)
    elif metric == "aecov":
        _, amplitude = _analytic_signal_matrices(epoch)
        matrix = np.cov(amplitude, bias=True)
    elif metric == "aecorr":
        _, amplitude = _analytic_signal_matrices(epoch)
        matrix = _safe_corrcoef(amplitude)
    elif metric == "plv":
        matrix = _plv_matrix(epoch)
    elif metric == "wplv":
        matrix = _wplv_matrix(epoch)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = stabilize_spd(_symmetrize(matrix))
    return matrix.astype(np.float64)


def compute_feature_set(data_dir: Path, cache_dir: Path, band: str, metric: str) -> FeatureSet:
    if band not in BANDS:
        raise ValueError(f"Unsupported band: {band}")
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{CACHE_VERSION}_{band}_{metric}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return FeatureSet(
            band=band,
            metric=metric,
            matrices=cached["matrices"],
            group_codes=cached["group_codes"],
            subject_ids=cached["subject_ids"],
        )

    matrices: list[np.ndarray] = []
    group_codes: list[str] = []
    subject_ids: list[str] = []

    for path in sorted(data_dir.glob(f"*_{band}-epo.fif")):
        stem = path.name.replace("-epo.fif", "")
        subject_id, group_code, _ = stem.split("_")
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        data = epochs.get_data(copy=False)
        sfreq = float(epochs.info["sfreq"])
        n_bins = freedman_diaconis_bins(data.ravel())
        for epoch in data:
            matrices.append(compute_connectivity_matrix(epoch, metric, sfreq, n_bins, band))
            group_codes.append(group_code)
            subject_ids.append(subject_id)

    feature_set = FeatureSet(
        band=band,
        metric=metric,
        matrices=np.stack(matrices, axis=0),
        group_codes=np.asarray(group_codes),
        subject_ids=np.asarray(subject_ids),
    )
    np.savez(
        cache_path,
        matrices=feature_set.matrices,
        group_codes=feature_set.group_codes,
        subject_ids=feature_set.subject_ids,
    )
    return feature_set
