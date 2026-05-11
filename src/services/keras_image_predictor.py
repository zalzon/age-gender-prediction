from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.services.prediction_result import PredictionResult


class KerasImagePredictionService:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or Path(__file__).resolve().parent.parent / "models"
        self._model_paths = {
            "MobileNetV2": self.models_dir / "mobilenetv2.h5",
            "NASNetMobile": self.models_dir / "nasnetmobile.h5",
        }
        self._loaded_models: dict[str, Any] = {}
        self.image_size = (224, 224)
        self.age_labels = ("child", "teen", "young_adult", "adult", "senior")
        self.age_group_ranges = {
            "child": "0-12",
            "teen": "13-19",
            "young_adult": "20-39",
            "adult": "40-59",
            "senior": "60+",
        }

    def predict(self, image_path: Path | str, selected_model: str) -> PredictionResult:
        model = self._load_model(selected_model)
        input_array = self._preprocess_image(image_path)

        predictions = model.predict(input_array, verbose=0)
        if isinstance(predictions, dict):
            gender_output = predictions["gender"]
            age_output = predictions["age"]
        else:
            gender_output, age_output = predictions

        female_probability = float(np.squeeze(gender_output))
        age_probabilities = np.squeeze(age_output)
        # expose raw probabilities for richer UI display
        self.last_gender_prob = female_probability
        self.last_age_probs = [float(x) for x in age_probabilities]
        age_index = int(np.argmax(age_probabilities))
        age_label = self.age_labels[age_index] if age_index < len(self.age_labels) else str(age_index)
        age_range = self.age_group_ranges.get(age_label, "Unknown")
        gender_label = "Female" if female_probability >= 0.5 else "Male"
        confidence_score = float(np.round(age_probabilities[age_index] * 100, 2))

        return PredictionResult(
            selected_model=selected_model,
            predicted_gender=gender_label,
            predicted_age_group=f"{age_label} ({age_range})",
            confidence_score=confidence_score,
        )

    def _load_model(self, selected_model: str) -> Any:
        if selected_model not in self._model_paths:
            raise ValueError(f"Unsupported model selected: {selected_model}")

        if selected_model in self._loaded_models:
            return self._loaded_models[selected_model]

        model_path = self._model_paths[selected_model]
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            from tensorflow.keras.models import load_model
        except Exception as exc:  # pragma: no cover - import availability depends on local environment
            raise RuntimeError(
                "TensorFlow/Keras is required to run the uploaded-image models. "
                "Install TensorFlow in a supported Python environment, then try again."
            ) from exc

        model = load_model(model_path, compile=False)
        self._loaded_models[selected_model] = model
        return model

    def _preprocess_image(self, image_path: Path | str) -> np.ndarray:
        image = Image.open(Path(image_path)).convert("RGB")
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        return np.expand_dims(image_array, axis=0)