from __future__ import annotations

from dataclasses import dataclass
from random import choice, uniform


@dataclass(frozen=True)
class PredictionResult:
    selected_model: str
    predicted_gender: str
    predicted_age_group: str
    confidence_score: float


class MockPredictionService:
    def __init__(self) -> None:
        self.gender_options = ("Male", "Female")
        self.age_group_options = (
            "0-2",
            "3-5",
            "6-10",
            "11-15",
            "16-20",
            "21-30",
            "31-40",
            "41-50",
            "51-60",
            "61+",
        )

    def predict(self, selected_model: str) -> PredictionResult:
        if selected_model not in {"MobileNetV2", "NASNetMobile"}:
            raise ValueError("Unsupported model selected.")

        if selected_model == "MobileNetV2":
            confidence = uniform(0.78, 0.96)
        else:
            confidence = uniform(0.75, 0.94)

        return PredictionResult(
            selected_model=selected_model,
            predicted_gender=choice(self.gender_options),
            predicted_age_group=choice(self.age_group_options),
            confidence_score=round(confidence * 100, 2),
        )
