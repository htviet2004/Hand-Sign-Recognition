import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import json

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Cấu hình ===
# Tự động nạp gesture_map.json nếu có (format: key -> {"label": ..., "desc": ..., ...})
GESTURE_MAP_FILE = "gesture_map.json"
if not os.path.exists(GESTURE_MAP_FILE):
    raise FileNotFoundError(f"Bắt buộc phải có {GESTURE_MAP_FILE} trong workspace. Hãy tạo file mapping gesture trước khi thu dữ liệu.")

try:
    with open(GESTURE_MAP_FILE, "r", encoding="utf-8") as f:
        gm = json.load(f)
    GESTURES = {}
    for k, v in gm.items():
        key = str(k)
        if isinstance(v, dict) and "label" in v:
            GESTURES[key] = v["label"]
        elif isinstance(v, str):
            GESTURES[key] = v
    if not GESTURES:
        raise ValueError(f"{GESTURE_MAP_FILE} không chứa mapping hợp lệ (key->label).")
    print(f"✅ Đã nạp gesture map từ {GESTURE_MAP_FILE}: {GESTURES}")
except Exception as e:
    raise RuntimeError(f"Lỗi khi đọc {GESTURE_MAP_FILE}: {e}")
NUM_SAMPLES = 300  # Số mẫu mỗi gesture

CSV_FILE = "data/my_gesture_landmarks.csv"
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

all_data = []
labels = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

stop_all = False  # Cho phép thoát an toàn thay vì exit()

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        for key, gesture_name in GESTURES.items():
            if stop_all:
                break
            print(f"👉 Bắt đầu thu gesture [{gesture_name}] — nhấn phím {key} để bắt đầu, 's' để bỏ qua, 'q' để thoát")

            # Đợi người dùng nhấn phím đúng để bắt đầu
            k = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Nhan phim {key} de bat dau, 's' de bo qua, 'q' de thoat", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Collecting Gestures", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(key):
                    break
                if k == ord('s'):
                    print(f"⏩ Bo qua gesture [{gesture_name}]")
                    break
                if k == ord('q'):
                    stop_all = True
                    print("🛑 Dung thu thap du lieu.")
                    break

            if stop_all:
                break
            if k is None or k == ord('s'):
                continue

            # Đếm ngược 3 giây
            for t in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"{gesture_name} trong {t}s",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imshow("Collecting Gestures", frame)
                # Cho phép nhấn 'q' để dừng trong lúc đếm
                k2 = cv2.waitKey(1) & 0xFF
                if k2 == ord('q'):
                    stop_all = True
                    print("🛑 Dung thu thap du lieu.")
                    break
                time.sleep(1)

            if stop_all:
                break

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
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Nhan 'q' de dung, 'n' de bo qua gesture nay",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    )

                cv2.imshow("Collecting Gestures", frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    stop_all = True
                    print("🛑 Dung thu thap du lieu.")
                    break
                if k == ord('n'):
                    print(f"⏩ Bo qua gesture [{gesture_name}]")
                    break

            if stop_all:
                break

    # Tạo DataFrame và lưu nếu có dữ liệu mới
    if len(all_data) > 0:
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

        # Loại bỏ trùng lặp (theo toàn bộ cột, bao gồm gesture_label)
        df_final = df_final.drop_duplicates()

        # Lưu CSV
        df_final.to_csv(CSV_FILE, index=False)
        print(f"✅ Đã ghi nối thêm dữ liệu (và loại bỏ trùng trong từng gesture) vào {CSV_FILE}")
    else:
        print("ℹ️ Khong co du lieu moi duoc thu thap, khong cap nhat file CSV.")
finally:
    cap.release()
    cv2.destroyAllWindows()
