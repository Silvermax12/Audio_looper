"""
Pydantic models for the Phrase-Song API.
"""

from typing import List, Optional
from pydantic import BaseModel


class LoopPair(BaseModel):
    """Represents a potential loop point pair with enhanced metrics."""
    start_time: float
    end_time: float
    duration: float
    similarity_score: float
    seamlessness_score: float  # 0-100 composite score
    quality_grade: str  # A, B, C, D
    beat_aligned: bool
    recommended_crossfade_ms: int
    bar_length: int  # Number of bars in this loop
    bar_category: str  # e.g., "Riff", "Phrase", "Section", "Verse"


class LoopAnalysisResponse(BaseModel):
    """Response containing loop analysis results."""
    loop_pairs: List[LoopPair]
    duration: float
    tempo: float
    beats_per_bar: int = 4
    key: Optional[str] = None
