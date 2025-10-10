% Hand Sign Recognition — Progress Report
% Your Name
% Date: 2025-09-24

---

# Vấn đề & Mục tiêu

- Vấn đề
  - Nhận dạng cử chỉ tay (hand gestures) cho ứng dụng tương tác.
  - Thách thức: biến thể góc, khoảng cách, chiếu sáng, tay trái/phải, và dữ liệu hạn chế.
- Mục tiêu
  - Xây dựng pipeline thu thập dữ liệu, tiền xử lý, và mô hình nhận dạng gesture.
  - Mục tiêu accuracy thử nghiệm: >= 85% trên tập test ban đầu.

> Speaker notes: Giới thiệu ngắn mục tiêu dự án, tại sao cần gesture recognition (ứng dụng thực tế).

---

# Dữ liệu

- Nguồn
  - Dữ liệu thu bằng webcam, sử dụng MediaPipe Hands để trích landmark (21 điểm x 3).
  - File dữ liệu: `data/my_gesture_landmarks.csv` (hoặc `data/` chứa artifacts).
- Kích thước hiện tại
  - Số gesture: (ví dụ) 13 classes (palm, fist, thumbs_up, ... none)
  - Mẫu/gesture: khuyến nghị 300, hiện có: xem `data/my_gesture_landmarks.csv`
- Metadata
  - Lưu scaler và label encoder: `gesture_scaler.joblib`, `gesture_label_encoder.joblib`

> Speaker notes: Nêu nguồn dữ liệu, format (63 features), và tập sample hiện tại.

---

# Xử lý dữ liệu

- Tiền xử lý
  - Loại bỏ dòng missing (dropna).
  - Chuẩn hóa tọa độ: translate theo cổ tay, scale theo kích thước palm (tùy chọn).
  - Chuẩn hóa bằng `StandardScaler`.
- Augmentation
  - Mirror (flip), gaussian jitter, small rotation/scale.
- Mapping labels
  - `gesture_map.json` quản lý mapping phím -> label + mô tả.

> Speaker notes: Giải thích các bước tiền xử lý, lý do chuẩn hóa và augmentation.

---

# Mô hình

- Mô hình hiện tại
  - RandomForestClassifier (n_estimators=200, class_weight='balanced')
- Lưu artifact
  - `gesture_model.joblib`, `gesture_scaler.joblib`, `gesture_label_encoder.joblib`
- Lộ trình cải tiến
  - Thử XGBoost/LightGBM hoặc MLP / Small CNN trên embedding
  - Cross-validation (5-fold) để đánh giá ổn định

> Speaker notes: Tại sao chọn RandomForest (đơn giản, không cần nhiều tuning), và các lựa chọn tiếp theo.

---

# Kết quả

- Đánh giá
  - Sử dụng stratified split 80/20.
  - Các metric: accuracy, classification report, confusion matrix.
- Demo realtime
  - `test.py` hiển thị nhãn realtime với smoothing, top-3 probabilities, và bar visualization.
- Điểm mạnh / hạn chế
  - Strengths: pipeline hoàn chỉnh, realtime demo, mapping dynamic.
  - Limitations: dữ liệu có thể chưa cân bằng, cần augmentation / thêm users.

> Speaker notes: Hiện tại chạy demo để minh họa (nếu live demo được). Đưa ra số liệu cụ thể nếu có (sửa slide này bằng các con số thực từ kết quả training).

---

# Next steps

- Thu thêm dữ liệu cho các lớp yếu
- Thử cross-validation và model alternatives
- Thêm CI để tự động train khi new data available
- Nâng UX realtime: cải thiện smoothing, threshold, edge detection

---

# Tài liệu & Code

- Repo: local workspace `HandSign/`
- Các file quan trọng: `train.py`, `train_gesture.py`, `test.py`, `gesture_map.json`, `data/*`

> Speaker notes: Cung cấp đường dẫn đến nguồn code và cách bắt đầu local.
