# Sign-to-Text Recognition

A real-time hand-gesture recognition project that uses MediaPipe hand landmarks and a Random Forest classifier to recognise a set of static hand signs and convert them into text. Optional text-to-speech is available through `pyttsx3`.

## Features

- Real-time hand tracking with OpenCV and MediaPipe
- Static-sign classification using a trained Random Forest model
- Hold-to-type behaviour to reduce accidental repeated predictions
- Clear, backspace, and optional speak controls
- Data collection and model-training modes

## Requirements

- Python 3.9 or a compatible Python version supported by your installed MediaPipe release
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python sign_to_text.py run
```

Keyboard controls in the camera window:

- `q` — quit
- `c` — clear the text
- `b` — backspace
- `s` — speak the current text (requires `pyttsx3`)

## Collect data and train

To collect samples for labels, for example:

```bash
python sign_to_text.py collect A B C
```

Show each sign to the camera and press `SPACE` when prompted. The script records samples for each label.

Train the classifier:

```bash
python sign_to_text.py train
```

The training script reports held-out accuracy and saves the trained model as `sign_model.joblib`.

## Files

- `sign_to_text.py` — data collection, training, and live recognition
- `signs_data.csv` — collected hand-landmark dataset
- `sign_model.joblib` — trained classifier

The MediaPipe hand-landmarker task model is downloaded automatically the first time the script runs; it is not stored in this repository.

## Notes

This project recognises the static labels represented in its included dataset. It is not a complete sign-language translation system and does not interpret full sentences, grammar, or dynamic signs.
