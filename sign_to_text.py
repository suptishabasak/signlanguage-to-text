"""
Sign language (static signs / ASL fingerspelling) -> text.

Three steps:
  1) Collect data:   python sign_to_text.py collect A B C D ...
       For each label, show the sign to the camera and press SPACE.
       It records 200 samples (move your hand slightly while recording).
       Use special labels  space  and  del  for a "space" sign and a "backspace" sign.
  2) Train:          python sign_to_text.py train
  3) Run live:       python sign_to_text.py run

Run keys:  q = quit   c = clear text   b = backspace   s = speak (needs pyttsx3)

Install:
  pip install opencv-python mediapipe numpy scikit-learn joblib
  pip install pyttsx3        # optional, for speaking the text
"""

import collections
import csv
import os
import sys
import urllib.request

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DATA_PATH = os.path.join(BASE, "signs_data.csv")
CLF_PATH = os.path.join(BASE, "sign_model.joblib")

SAMPLES_PER_LABEL = 200
CONF_THRESHOLD = 0.70   # min classifier confidence
HOLD_FRAMES = 15        # frames a sign must be held to be typed (~0.5 s)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

if not os.path.exists(MODEL_PATH):
    print("Downloading hand model (one time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# ---------------- helpers ----------------
def make_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def features(lm):
    """21 landmarks -> 63 numbers, independent of hand position and size."""
    a = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    a -= a[0]                                   # wrist = origin
    scale = np.abs(a[:, :2]).max()
    if scale > 0:
        a /= scale
    return a.flatten()


def draw_hand(frame, lm):
    h, w, _ = frame.shape
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (255, 0, 255), cv2.FILLED)


def detect(landmarker, frame, ts_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(img, ts_ms)
    return res.hand_landmarks[0] if res.hand_landmarks else None


def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    return cap


# ---------------- 1) collect ----------------
def collect(labels):
    if not labels:
        print("Usage: python sign_to_text.py collect A B C ...")
        return
    landmarker = make_landmarker()
    cap = open_camera()
    ts = 0
    idx = 0
    recording = 0

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        while idx < len(labels):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            ts += 33
            lm = detect(landmarker, frame, ts)
            label = labels[idx]

            if lm:
                draw_hand(frame, lm)
                if recording > 0:
                    writer.writerow([label] + features(lm).tolist())
                    recording -= 1
                    if recording == 0:
                        f.flush()
                        idx += 1
                        continue

            if recording > 0:
                msg = f"Recording '{label}': {SAMPLES_PER_LABEL - recording}/{SAMPLES_PER_LABEL}"
            else:
                msg = f"Show '{label}' then press SPACE  ({idx + 1}/{len(labels)})"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Collect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and recording == 0 and lm:
                recording = SAMPLES_PER_LABEL

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {DATA_PATH}. Now run: python sign_to_text.py train")


# ---------------- 2) train ----------------
def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    if not os.path.exists(DATA_PATH):
        print("No data yet. Run collect first.")
        return
    X, y = [], []
    with open(DATA_PATH) as f:
        for row in csv.reader(f):
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    X, y = np.array(X), np.array(y)
    print(f"{len(X)} samples, {len(set(y))} classes: {sorted(set(y))}")
    if len(set(y)) < 2:
        print("Need at least 2 different labels.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print(f"Held-out accuracy: {clf.score(X_te, y_te) * 100:.1f}%")

    clf.fit(X, y)  # final model uses all data
    joblib.dump(clf, CLF_PATH)
    print(f"Model saved. Now run: python sign_to_text.py run")


# ---------------- 3) run ----------------
def run():
    if not os.path.exists(CLF_PATH):
        print("No trained model. Run collect, then train.")
        return
    clf = joblib.load(CLF_PATH)
    classes = clf.classes_

    speaker = None
    try:
        import pyttsx3
        speaker = pyttsx3.init()
    except Exception:
        pass

    landmarker = make_landmarker()
    cap = open_camera()
    ts = 0
    text = ""
    holding, hold, locked = None, 0, None
    recent = collections.deque(maxlen=5)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        ts += 33
        lm = detect(landmarker, frame, ts)

        label, conf = None, 0.0
        if lm:
            draw_hand(frame, lm)
            proba = clf.predict_proba([features(lm)])[0]
            i = int(np.argmax(proba))
            conf = float(proba[i])
            if conf >= CONF_THRESHOLD:
                recent.append(classes[i])
                # light smoothing: most common of last few predictions
                label = collections.Counter(recent).most_common(1)[0][0]
            else:
                recent.clear()
        else:
            recent.clear()
            locked = None  # hand left: allows repeating the same letter

        # hold-to-type logic
        if label and label == holding:
            hold += 1
        else:
            holding, hold = label, (1 if label else 0)
            if label and label != locked:
                locked = None

        if holding and hold >= HOLD_FRAMES and holding != locked:
            if holding == "space":
                text += " "
            elif holding == "del":
                text = text[:-1]
            else:
                text += holding
            locked = holding

        # HUD
        if label:
            cv2.putText(frame, f"{label} ({conf * 100:.0f}%)", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            bar = int(min(hold / HOLD_FRAMES, 1.0) * 200)
            cv2.rectangle(frame, (10, 50), (210, 65), (80, 80, 80), 2)
            cv2.rectangle(frame, (10, 50), (10 + bar, 65), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, text[-35:] + "_", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Sign to Text", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            text = ""
        elif key == ord("b"):
            text = text[:-1]
        elif key == ord("s") and speaker and text.strip():
            speaker.say(text)
            speaker.runAndWait()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "collect":
        collect(sys.argv[2:])
    elif cmd == "train":
        train()
    elif cmd == "run":
        run()
    else:
        print(__doc__)