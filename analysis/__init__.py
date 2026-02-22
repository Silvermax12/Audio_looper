"""
Audio analysis sub-package.
Contains spectral analysis, loop detection, and audio resize utilities.
"""

from .loop_detection import detect_seamless_loops, apply_crossfade, snap_to_beat, is_beat_aligned
from .audio_resize import shorten_audio_intelligent, lengthen_audio_intelligent, detect_segment_boundaries
from .spectral import (
    compute_spectral_flux, compute_rms_energy,
    compute_boundary_similarity, compute_seamlessness_score,
    get_quality_grade, compute_recurrence_matrix
)

__all__ = [
    'detect_seamless_loops', 'apply_crossfade', 'snap_to_beat', 'is_beat_aligned',
    'shorten_audio_intelligent', 'lengthen_audio_intelligent', 'detect_segment_boundaries',
    'compute_spectral_flux', 'compute_rms_energy',
    'compute_boundary_similarity', 'compute_seamlessness_score',
    'get_quality_grade', 'compute_recurrence_matrix',
]
