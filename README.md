# Age & Gender Prediction Dashboard

This repository is the GUI/frontend for our age-group and gender prediction project. The current build is a working prototype: the interface is finished enough for testing, but the actual ML models are still mocked so the team can keep iterating on layout, flow, and integration without blocking on model files.

## Current Status

- Tkinter + ttk dashboard is implemented.
- Image upload and preview work.
- Model selection is available for `MobileNetV2` and `NASNetMobile`.
- Prediction output is currently mock data from `src/services/mock_predictor.py`.
- Live webcam analysis is integrated in the UI and updates the preview/result panel.
- The webcam flow is intentionally conservative on frame updates, so it may not run at full FPS on every machine.

## What Is Implemented

- `src/gui/main_window.py` contains the full dashboard layout.
- `src/main.py` is the module entry point.
- `app.py` is the simple launcher.
- `src/services/mock_predictor.py` returns placeholder gender, age group, confidence, and selected model values.
- `requirements.txt` includes the UI and webcam dependencies, including `Pillow` and `opencv-python`.

## Project Structure

```text
AgeGenderPrediction/
├── app.py
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── gui/
    │   └── main_window.py
    └── services/
        └── mock_predictor.py
```

## How To Run

1. Create or activate your virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python app.py
   ```

You can also run the module directly:

```bash
python -m src.main
```

## Notes For The Team

- When we add real models, we should place them in a dedicated `models/` folder and document the exact file name and preprocessing steps.
- The current UI is designed so uploaded-image mode and live-camera mode share the same results panel.
- The live mode is functional, but it is still a prototype and may need threading or lower update intervals if we want smoother FPS.

## Next Implementation Step

The next major backend step is to replace `MockPredictionService` with a real model loader so the GUI can call a stable prediction interface regardless of which model is used.
