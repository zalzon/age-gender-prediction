# Age & Gender Prediction

A desktop application that predicts age group and gender from uploaded images using Keras models.

## Features

- Image upload and preview
- Two model options: MobileNetV2 and NASNetMobile
- Displays age group and gender predictions

## Project Structure

```
AgeGenderPrediction/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── main.py
    ├── gui/
    │   └── main_window.py
    ├── models/
    │   ├── mobilenetv2.h5
    │   └── nasnetmobile.h5
    └── services/
        ├── keras_image_predictor.py
        └── prediction_result.py
```

## Requirements

- Python 3.10 or higher
- See `requirements.txt` for dependencies

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the application:
   ```bash
   python app.py
   ```

## How to Use

1. Upload an image (224x224 RGB format recommended)
2. Select a model (MobileNetV2 or NASNetMobile)
3. View the age group and gender predictions
