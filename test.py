import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load model, scaler, encoder
model = joblib.load("data/gesture_model.pkl")
scaler = joblib.load("data/scaler.pkl")
le = joblib.load("data/label_encoder.pkl")

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        # Vẽ landmarks với style đẹp hơn
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),   # Landmarks: đỏ, to
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)    # Connections: xanh lá
        )

        # Trích landmark
        data_point = []
        for lm in hand_landmarks.landmark:
            data_point.extend([lm.x, lm.y, lm.z])

        X = np.array(data_point).reshape(1, -1)
        X = scaler.transform(X)

        pred = model.predict(X)[0]
        gesture_name = le.inverse_transform([pred])[0]

        # Lấy xác suất dự đoán
        probs = model.predict_proba(X)[0]
        confidence = np.max(probs) * 100

        cv2.putText(frame, f"Gesture: {gesture_name} ({confidence:.1f}%)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Real-time Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
