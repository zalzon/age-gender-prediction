# Age & Gender Prediction Dashboard

This repository is the GUI/frontend for our age-group and gender prediction project. The current build is a working prototype: the interface is finished enough for testing, and the uploaded-image path now uses the saved Keras models from `src/models/`.

## Current Status

- Tkinter + ttk dashboard is implemented.
- Image upload and preview work.
- Model selection is available for `MobileNetV2` and `NASNetMobile`.
- Upload-image predictions use `src/models/mobilenetv2.h5` and `src/models/nasnetmobile.h5`.
- Live webcam analysis is still separate and remains on the mock path for now.
- The webcam flow is intentionally conservative on frame updates, so it may not run at full FPS on every machine.

## What Is Implemented

- `src/gui/main_window.py` contains the full dashboard layout.
- `src/main.py` is the module entry point.
- `app.py` is the simple launcher.
- `src/services/keras_image_predictor.py` loads the saved Keras models for uploaded-image prediction.
- `src/services/mock_predictor.py` remains available for the live webcam placeholder flow.
- `requirements.txt` includes the UI, webcam, and model-loading dependencies.

## Project Structure

```text
AgeGenderPrediction/
app.py
README.md
requirements.txt
src/main.py
src/models/mobilenetv2.h5
src/models/nasnetmobile.h5
src/gui/main_window.py
src/services/keras_image_predictor.py
src/services/mock_predictor.py
```

## How To Run

1. Create or activate your virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The upload-image predictor needs TensorFlow. On Python 3.14, pip will skip TensorFlow because no compatible wheel is available yet. Use Python 3.10-3.12 to enable real model inference.

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
- The uploaded-image models expect 224x224 RGB input and return two outputs: gender and age group.
- The current UI is designed so uploaded-image mode and live-camera mode share the same results panel.
- The live mode is functional, but it is still a prototype and may need threading or lower update intervals if we want smoother FPS.

## Next Implementation Step

The next major backend step is to replace `MockPredictionService` with a real model loader so the GUI can call a stable prediction interface regardless of which model is used.
