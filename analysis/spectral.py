"""
Spectral analysis utilities for audio loop detection.
Handles spectral flux, RMS energy, boundary similarity, seamlessness scoring,
quality grading, and recurrence matrix computation.
"""

import numpy as np
import librosa


def compute_spectral_flux(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Compute spectral flux — measures rate of spectral change."""
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    diff = np.diff(S, axis=1)
    diff = np.maximum(0, diff)
    flux = np.sum(diff, axis=0)
    flux = np.concatenate([[0], flux])
    return flux


def compute_rms_energy(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Compute RMS energy per frame."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return rms


def compute_boundary_similarity(
    y: np.ndarray,
    sr: int,
    start_sample: int,
    end_sample: int,
    window_samples: int = 2048
) -> dict:
    """
    Compute similarity metrics at loop boundaries.
    Compares the region around the end with the region around the start.
    """
    end_region = y[max(0, end_sample - window_samples):end_sample]
    start_region = y[start_sample:min(len(y), start_sample + window_samples)]

    min_len = min(len(end_region), len(start_region))
    if min_len < 100:
        return {'correlation': 0.5, 'energy_match': 0.5, 'spectral_match': 0.5}

    end_region = end_region[:min_len]
    start_region = start_region[:min_len]

    # Cross-correlation (normalized)
    correlation = np.corrcoef(end_region, start_region)[0, 1]
    if np.isnan(correlation):
        correlation = 0.5
    correlation = (correlation + 1) / 2  # Normalize to 0-1

    # RMS Energy match
    rms_end = np.sqrt(np.mean(end_region ** 2))
    rms_start = np.sqrt(np.mean(start_region ** 2))
    if rms_end + rms_start > 0:
        energy_match = 1 - abs(rms_end - rms_start) / (rms_end + rms_start)
    else:
        energy_match = 1.0

    # Spectral similarity (using short-time FFT)
    spec_end = np.abs(np.fft.rfft(end_region * np.hanning(min_len)))
    spec_start = np.abs(np.fft.rfft(start_region * np.hanning(min_len)))

    spec_end = spec_end / (np.linalg.norm(spec_end) + 1e-8)
    spec_start = spec_start / (np.linalg.norm(spec_start) + 1e-8)

    spectral_match = np.dot(spec_end, spec_start)

    return {
        'correlation': float(correlation),
        'energy_match': float(energy_match),
        'spectral_match': float(spectral_match)
    }


def compute_seamlessness_score(
    y: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    chroma_similarity: float
) -> tuple[float, int]:
    """
    Compute composite seamlessness score (0-100) and recommended crossfade.

    Weights:
    - Chroma similarity: 30%
    - Energy match: 25%
    - Spectral match: 25%
    - Cross-correlation: 20%
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    window_samples = int(0.05 * sr)  # ~50ms

    boundary_metrics = compute_boundary_similarity(
        y, sr, start_sample, end_sample, window_samples
    )

    score = (
        0.30 * chroma_similarity +
        0.25 * boundary_metrics['energy_match'] +
        0.25 * boundary_metrics['spectral_match'] +
        0.20 * boundary_metrics['correlation']
    )

    seamlessness = min(100, max(0, score * 100))

    if seamlessness >= 85:
        crossfade_ms = 20
    elif seamlessness >= 70:
        crossfade_ms = 50
    elif seamlessness >= 50:
        crossfade_ms = 100
    else:
        crossfade_ms = 150

    return seamlessness, crossfade_ms


def get_quality_grade(seamlessness: float) -> str:
    """Convert seamlessness score to letter grade."""
    if seamlessness >= 85:
        return "A"
    elif seamlessness >= 70:
        return "B"
    elif seamlessness >= 55:
        return "C"
    else:
        return "D"


def compute_recurrence_matrix(chroma: np.ndarray) -> np.ndarray:
    """Compute Recurrence Matrix using chroma features with affinity mode."""
    rec_matrix = librosa.segment.recurrence_matrix(
        chroma,
        mode='affinity',
        metric='cosine',
        sparse=False
    )
    return rec_matrix
