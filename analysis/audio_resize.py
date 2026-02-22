"""
Audio resize utilities — intelligent shortening and lengthening.
Uses segment boundary detection and energy-based selection.
"""

import numpy as np
import librosa

from .loop_detection import detect_seamless_loops, apply_crossfade


def detect_segment_boundaries(
    y: np.ndarray,
    sr: int,
    n_segments: int = 10
) -> np.ndarray:
    """
    Detect natural segment boundaries in audio using spectral clustering.
    Returns timestamps of segment boundaries.
    """
    hop_length = 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    try:
        bounds = librosa.segment.agglomerative(mfcc, n_segments)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)
    except Exception:
        duration = len(y) / sr
        bound_times = np.linspace(0, duration, n_segments + 1)

    return bound_times


def shorten_audio_intelligent(
    y: np.ndarray,
    sr: int,
    target_duration: float,
    preserve_start: bool = True,
    preserve_end: bool = True
) -> np.ndarray:
    """
    Intelligently shorten audio to target duration.
    Finds natural segment boundaries and removes least important sections.
    """
    current_duration = len(y) / sr

    if current_duration <= target_duration:
        return y

    remove_duration = current_duration - target_duration

    n_segments = max(5, int(current_duration / 5))
    boundaries = detect_segment_boundaries(y, sr, n_segments)

    segments_info = []
    for i in range(len(boundaries) - 1):
        start_sample = int(boundaries[i] * sr)
        end_sample = int(boundaries[i + 1] * sr)
        segment = y[start_sample:end_sample]

        if preserve_start and i == 0:
            continue
        if preserve_end and i == len(boundaries) - 2:
            continue

        rms = np.sqrt(np.mean(segment ** 2))
        segments_info.append({
            'index': i,
            'start': boundaries[i],
            'end': boundaries[i + 1],
            'duration': boundaries[i + 1] - boundaries[i],
            'energy': rms
        })

    # Remove lowest energy segments first
    segments_info.sort(key=lambda x: x['energy'])

    removed_duration = 0
    segments_to_remove = set()

    for seg in segments_info:
        if removed_duration >= remove_duration:
            break
        segments_to_remove.add(seg['index'])
        removed_duration += seg['duration']

    # Reconstruct without removed segments
    result_parts = []
    for i in range(len(boundaries) - 1):
        if i not in segments_to_remove:
            start_sample = int(boundaries[i] * sr)
            end_sample = int(boundaries[i + 1] * sr)
            result_parts.append(y[start_sample:end_sample])

    if len(result_parts) == 0:
        return y[:int(target_duration * sr)]

    # Crossfade between segments (20ms)
    crossfade_samples = int(0.02 * sr)
    result = result_parts[0]

    for part in result_parts[1:]:
        if len(result) > crossfade_samples and len(part) > crossfade_samples:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)

            result[-crossfade_samples:] = (
                result[-crossfade_samples:] * fade_out +
                part[:crossfade_samples] * fade_in
            )
            result = np.concatenate([result, part[crossfade_samples:]])
        else:
            result = np.concatenate([result, part])

    return result


def lengthen_audio_intelligent(
    y: np.ndarray,
    sr: int,
    target_duration: float,
    loop_start: float = None,
    loop_end: float = None
) -> np.ndarray:
    """
    Intelligently lengthen audio by looping sections.
    """
    current_duration = len(y) / sr

    if current_duration >= target_duration:
        return y

    # Find best loopable section if not specified
    if loop_start is None or loop_end is None:
        loop_pairs, _, _ = detect_seamless_loops(
            y, sr, top_n=1,
            min_loop_duration=5.0,
            max_loop_duration=min(15.0, current_duration * 0.5)
        )

        if loop_pairs:
            loop_start = loop_pairs[0].start_time
            loop_end = loop_pairs[0].end_time
        else:
            loop_start = current_duration * 0.3
            loop_end = current_duration * 0.7

    loop_duration = loop_end - loop_start
    needed_duration = target_duration - current_duration
    n_loops = int(np.ceil(needed_duration / loop_duration))

    start_sample = int(loop_start * sr)
    end_sample = int(loop_end * sr)

    # Apply crossfade to loop section (50ms)
    crossfade_samples = int(0.05 * sr)
    loop_section = apply_crossfade(y, sr, start_sample, end_sample, crossfade_samples)

    # Build: [before loop] + [loop repeated] + [after loop]
    before = y[:start_sample]
    after = y[end_sample:]

    repeated = np.tile(loop_section, n_loops + 1)
    result = np.concatenate([before, repeated, after])
    target_samples = int(target_duration * sr)

    if len(result) > target_samples:
        result = result[:target_samples]

    return result
