import cv2
import mediapipe as mp
import pandas as pd
import os
import time

mp_hands = mp.solutions.hands

# === Cấu hình ===
GESTURES = {
    "0": "palm",
    # "1": "fist",
    # "2": "thumbs_up",
    # "3": "ok",
    # "4": "peace",
    # "5": "rock",
    # "6": "call",
}
NUM_SAMPLES = 300  # Số mẫu mỗi gesture

CSV_FILE = "data/my_gesture_landmarks.csv"
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

all_data = []
labels = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
) as hands:

    for key, gesture_name in GESTURES.items():
        print(f"👉 Bắt đầu thu gesture [{gesture_name}] — nhấn phím {key} để bắt đầu hoặc 's' để skip")

        # Đợi người dùng nhấn phím đúng để bắt đầu
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Nhan phim {key} de bat dau, 's' de bo qua", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Collecting Gestures", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(key):
                break
            if k == ord('s'):
                print(f"⏩ Bo qua gesture [{gesture_name}]")
                break

        if k == ord('s'):
            continue

        # Đếm ngược 3 giây
        for t in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{gesture_name} trong {t}s",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            cv2.imshow("Collecting Gestures", frame)
            cv2.waitKey(1)
            time.sleep(1)

        collected = 0
        last_save_time = time.time()

        while collected < NUM_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            cv2.putText(frame, f"Thu {gesture_name}: {collected}/{NUM_SAMPLES}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, "Nhan 'q' de dung, 'n' de bo qua gesture nay",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                # Chỉ lấy mẫu mỗi 0.1s để giảm nhiễu
                if time.time() - last_save_time > 0.1:
                    data_point = []
                    for lm in hand_landmarks.landmark:
                        data_point.extend([lm.x, lm.y, lm.z])
                    # Đảm bảo đủ 21 landmark
                    if len(data_point) == 63:
                        all_data.append(data_point)
                        labels.append(gesture_name)
                        collected += 1
                        last_save_time = time.time()

                # Vẽ landmarks đẹp hơn
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=4),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                )

            cv2.imshow("Collecting Gestures", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print("🛑 Dung thu thap du lieu.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            if k == ord('n'):
                print(f"⏩ Bo qua gesture [{gesture_name}]")
                break

# Tạo DataFrame mới
columns = []
for i in range(21):
    columns.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])

df_new = pd.DataFrame(all_data, columns=columns)
df_new["gesture_label"] = labels

# Nếu file đã tồn tại thì nối thêm
if os.path.exists(CSV_FILE):
    df_old = pd.read_csv(CSV_FILE)
    df_final = pd.concat([df_old, df_new], ignore_index=True)
else:
    df_final = df_new

# Loại bỏ trùng lặp (theo landmark + gesture_label)
df_final = df_final.drop_duplicates(subset=df_final.columns.tolist())

# Lưu CSV
df_final.to_csv(CSV_FILE, index=False)
print(f"✅ Đã ghi nối thêm dữ liệu (và loại bỏ trùng trong từng gesture) vào {CSV_FILE}")

cap.release()
cv2.destroyAllWindows()
