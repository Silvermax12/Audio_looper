"""
Seamless loop detection using beat-synchronous analysis.
Finds optimal loop points in audio using chroma features and beat tracking.
"""

import sys
import os
from typing import List

import numpy as np
import librosa

# Add parent directory to path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import LoopPair

from .spectral import (
    compute_recurrence_matrix,
    compute_seamlessness_score,
    get_quality_grade,
)


def snap_to_beat(time: float, beat_times: np.ndarray) -> float:
    """Snap a time to the nearest beat position."""
    if len(beat_times) == 0:
        return time
    idx = np.argmin(np.abs(beat_times - time))
    return float(beat_times[idx])


def is_beat_aligned(time: float, beat_times: np.ndarray, tolerance: float = 0.05) -> bool:
    """Check if a time is close to a beat position."""
    if len(beat_times) == 0:
        return False
    return np.min(np.abs(beat_times - time)) < tolerance


def apply_crossfade(
    y: np.ndarray,
    sr: int,
    start_sample: int,
    end_sample: int,
    crossfade_samples: int
) -> np.ndarray:
    """
    Apply crossfade to create seamless loop.
    Blends the end of the segment into the beginning.
    """
    segment = y[start_sample:end_sample].copy()

    if crossfade_samples <= 0 or len(segment) < crossfade_samples * 2:
        return segment

    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)

    crossfade_region = (
        segment[:crossfade_samples] * fade_in +
        segment[-crossfade_samples:] * fade_out
    )

    result = segment.copy()
    result[:crossfade_samples] = crossfade_region
    result = result[:-crossfade_samples]

    return result


def detect_seamless_loops(
    y: np.ndarray,
    sr: int,
    top_n: int = 5,
    min_loop_duration: float = 3.0,
    max_loop_duration: float = 30.0,
    similarity_threshold: float = 0.6
) -> tuple[List[LoopPair], float, float]:
    """
    Detect seamless loops using Beat-Synchronous Analysis.

    This approach:
    1. Uses librosa.beat.beat_track to find all beat timestamps
    2. Uses librosa.util.sync to aggregate chroma features to beat frames
    3. Only compares beat-to-beat similarities
    4. Ensures every loop starts/ends exactly on a downbeat
    """
    hop_length = 512
    duration = librosa.get_duration(y=y, sr=sr)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0])

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    n_beats = len(beat_times)

    if n_beats < 4:
        return [], duration, tempo

    # Chroma features aggregated to beats.
    # chroma_stft is used instead of chroma_cqt: both produce 12-bin chroma
    # but stft-based avoids the large CQT filterbank, saving ~60% RAM.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    # Free the full-resolution chroma — we only need the beat-synced version.
    del chroma
    beat_rec_matrix = compute_recurrence_matrix(beat_chroma)

    # Beat constraints
    avg_beat_duration = np.mean(np.diff(beat_times)) if n_beats > 1 else 0.5
    min_beats = max(2, int(min_loop_duration / avg_beat_duration))
    max_beats = min(n_beats - 1, int(max_loop_duration / avg_beat_duration))

    candidates = []

    # Search for high similarity beat pairs
    for i in range(n_beats):
        for j in range(i + min_beats, min(i + max_beats + 1, n_beats)):
            similarity = beat_rec_matrix[i, j]

            if similarity > similarity_threshold:
                # Beat-Phase Alignment: ensure same phase (index % 4)
                if i % 4 != j % 4:
                    continue

                start_time = beat_times[i]
                end_time = beat_times[j]
                loop_duration = end_time - start_time

                if not (min_loop_duration <= loop_duration <= max_loop_duration):
                    continue

                seamlessness, crossfade_ms = compute_seamlessness_score(
                    y, sr, start_time, end_time, similarity
                )

                quality_grade = get_quality_grade(seamlessness)

                candidates.append({
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'duration': float(loop_duration),
                    'similarity_score': float(similarity),
                    'seamlessness_score': float(seamlessness),
                    'quality_grade': quality_grade,
                    'adjusted_score': float(seamlessness),
                    'beat_aligned': True,
                    'crossfade_ms': crossfade_ms,
                    'start_beat': i,
                    'end_beat': j,
                    'n_beats': j - i
                })

    # Sort by seamlessness score
    candidates.sort(key=lambda x: x['adjusted_score'], reverse=True)

    # Functional Categories (Time-based)
    functional_categories = {
        'Riff': (4.0, 7.0),
        'Phrase': (8.0, 13.0),
        'Section': (16.0, 26.0),
        'Verse': (30.0, 999.0)
    }

    best_per_category = {}

    for c in candidates:
        dur = c['duration']
        for cat_name, (min_dur, max_dur) in functional_categories.items():
            if min_dur <= dur <= max_dur:
                if cat_name not in best_per_category:
                    c['bar_category'] = cat_name
                    c['bar_length'] = round(c['n_beats'] / 4)
                    best_per_category[cat_name] = c
                break

    # Collect results in order
    filtered = []
    for cat in ['Riff', 'Phrase', 'Section', 'Verse']:
        if cat in best_per_category:
            filtered.append(best_per_category[cat])

    # Convert to LoopPair objects
    loop_pairs = [
        LoopPair(
            start_time=round(c['start_time'], 3),
            end_time=round(c['end_time'], 3),
            duration=round(c['duration'], 3),
            similarity_score=round(c['similarity_score'], 4),
            seamlessness_score=round(c['seamlessness_score'], 1),
            quality_grade=c['quality_grade'],
            beat_aligned=True,
            recommended_crossfade_ms=c['crossfade_ms'],
            bar_length=c['bar_length'],
            bar_category=c['bar_category']
        )
        for c in filtered
    ]

    return loop_pairs, duration, tempo
