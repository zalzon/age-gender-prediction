from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    selected_model: str
    predicted_gender: str
    predicted_age_group: str
    confidence_score: float
