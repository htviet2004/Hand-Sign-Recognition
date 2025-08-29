import cv2
import joblib
import numpy as np
import pyautogui
import mediapipe as mp
import keyboard
import time
import os
import sys

# Đường dẫn tới model và scaler
MODEL_PATH = "data/gesture_model.pkl"
SCALER_PATH = "data/scaler.pkl"
ENCODER_PATH = "data/label_encoder.pkl"

# Kiểm tra file tồn tại
for path in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

# Load model, scaler, encoder
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Gesture to action mapping (bổ sung hành động tại đây)
GESTURE_ACTIONS = {
    # "gesture_label": lambda: print("Action for gesture_label"),
    # Ví dụ: "fist": lambda: keyboard.press_and_release('space')
}

last_gesture = None
last_action_time = 0
COOLDOWN = 1.0  # giây

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
screen_w, screen_h = pyautogui.size()

def process_hand_landmarks(hand_landmarks):
    """Chuyển landmarks thành vector đặc trưng."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def main():
    global last_gesture, last_action_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể truy cập webcam.")
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = process_hand_landmarks(hand_landmarks)

                # Lấy tọa độ ngón trỏ (landmark 8)
                index_finger = hand_landmarks.landmark[8]
                mouse_x = int(screen_w * index_finger.x)
                mouse_y = int(screen_h * index_finger.y)
                pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)

                # Dự đoán gesture
                X = scaler.transform([landmarks])
                pred = model.predict(X)[0]
                gesture_name = le.inverse_transform([pred])[0]
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Thực hiện hành động nếu gesture thay đổi và đủ thời gian cooldown
                current_time = time.time()
                if (gesture_name != last_gesture) or (current_time - last_action_time > COOLDOWN):
                    action = GESTURE_ACTIONS.get(gesture_name)
                    if action:
                        action()
                        last_action_time = current_time
                    last_gesture = gesture_name

        cv2.imshow("Hand Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()