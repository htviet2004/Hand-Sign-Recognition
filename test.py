import os
import cv2
import mediapipe as mp
import joblib
import json
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Load model, scaler, encoder
DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "gesture_model.joblib")
SCALER_PATH = os.path.join(DATA_DIR, "gesture_scaler.joblib")
ENCODER_PATH = os.path.join(DATA_DIR, "gesture_label_encoder.joblib")

for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Khong tim thay '{p}'. Hay train mo hinh truoc bang train_gesture.py"
        )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)
label_names = list(le.classes_)

# Optional: load gesture descriptions from gesture_map.json if present
LABEL_DESCRIPTIONS = {}
GM_PATH = "gesture_map.json"
if os.path.exists(GM_PATH):
    try:
        with open(GM_PATH, "r", encoding="utf-8") as f:
            gm = json.load(f)
        for k, v in gm.items():
            if isinstance(v, dict) and "label" in v:
                LABEL_DESCRIPTIONS[v["label"]] = v.get("desc")
    except Exception:
        # ignore errors and continue without descriptions
        LABEL_DESCRIPTIONS = {}

# Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Smoothing: trung binh xac suat tren N frame gan nhat
SMOOTH_WIN = 5
UNKNOWN_THRESHOLD = 0.4  # Neu max probability < threshold -> hien 'Unknown'
proba_buffer = deque(maxlen=SMOOTH_WIN)

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            display_text = "Nhan 'q' de thoat"
            top_info = None  # reset moi frame de tranh giu gia tri cu

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                # Vẽ landmarks với style đẹp hơn
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                )

                # Trích landmark
                data_point = []
                for lm in hand_landmarks.landmark:
                    data_point.extend([lm.x, lm.y, lm.z])

                if len(data_point) == 63:
                    X = np.array(data_point, dtype=np.float32).reshape(1, -1)
                    X = scaler.transform(X)

                    # Du doan
                    pred = model.predict(X)[0]
                    gesture_name = le.inverse_transform([pred])[0]

                    # Xac suat
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)[0]
                        proba_buffer.append(probs)
                        probs_smoothed = np.mean(proba_buffer, axis=0)
                        max_proba = float(np.max(probs_smoothed))
                        confidence = max_proba * 100

                        # Top-3 (smoothed)
                        top_idx = np.argsort(probs_smoothed)[::-1][:3]
                        top_info = [
                            (label_names[i], float(probs_smoothed[i])) for i in top_idx
                        ]

                        # Use top-1 from smoothed probabilities for display (respect Unknown threshold)
                        top1_label = label_names[top_idx[0]] if len(top_idx) > 0 else gesture_name
                        if max_proba < UNKNOWN_THRESHOLD:
                            gesture_name_display = "Unknown"
                        else:
                            gesture_name_display = top1_label
                    else:
                        probs_smoothed = None
                        confidence = 100.0  # Khong co predict_proba
                        gesture_name_display = gesture_name
                        top_info = None

                    display_text = f"Gesture: {gesture_name_display} ({confidence:.1f}%)"
                else:
                    display_text = "Khong du landmark (63)"

            # Overlay text huong dan/ket qua
            cv2.putText(
                frame,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            # If we have a short description for this label, show it below
            desc = None
            if 'gesture_name_display' in locals():
                desc = LABEL_DESCRIPTIONS.get(gesture_name_display)
            if desc:
                cv2.putText(
                    frame,
                    desc,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            # Ve top-3 proba: thanh ngang + text
            if top_info:
                # Config
                BAR_X = 10
                BAR_Y_START = 110
                BAR_WIDTH = 300
                BAR_HEIGHT = 18
                BAR_GAP = 10
                COLOR_BG = (40, 40, 40)
                COLORS = [(0, 255, 255), (0, 200, 255), (0, 150, 255)]  # Cyan tones

                y = BAR_Y_START
                for idx, (name, p) in enumerate(top_info):
                    w = int(BAR_WIDTH * max(0.0, min(1.0, p)))
                    # Nen bar
                    cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_WIDTH, y + BAR_HEIGHT), COLOR_BG, thickness=-1)
                    # Fill theo xac suat
                    cv2.rectangle(frame, (BAR_X, y), (BAR_X + w, y + BAR_HEIGHT), COLORS[idx % len(COLORS)], thickness=-1)
                    # Vien
                    cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_WIDTH, y + BAR_HEIGHT), (200, 200, 200), thickness=1)
                    # Text
                    txt = f"{name}: {p*100:.1f}%"
                    cv2.putText(frame, txt, (BAR_X + BAR_WIDTH + 10, y + BAR_HEIGHT - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y += BAR_HEIGHT + BAR_GAP

            cv2.imshow("Real-time Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
