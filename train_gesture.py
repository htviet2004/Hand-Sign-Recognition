import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

np.random.seed(42)

DATA_CSV = "data/my_gesture_landmarks.csv"
SAVE_DIR = os.path.dirname(DATA_CSV) or "."
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== LOAD DATA ====
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(
        f"Khong tim thay file du lieu: {DATA_CSV}. Hay chay script thu thap du lieu truoc (train.py)."
    )

df = pd.read_csv(DATA_CSV)

# Kiểm tra và loại bỏ dòng thiếu dữ liệu
df = df.dropna()
print(f"Số mẫu sau khi loại bỏ thiếu dữ liệu: {len(df)}")

if "gesture_label" not in df.columns:
    raise KeyError("Cot 'gesture_label' khong ton tai trong CSV. Hay kiem tra lai file du lieu.")

X = df.drop("gesture_label", axis=1)
y = df["gesture_label"]

# Thong ke phan bo lop
cls_counts = y.value_counts().sort_index()
print("Phan bo lop:")
for cls, cnt in cls_counts.items():
    print(f"  - {cls}: {cnt}")

# Encode label thành số
le = LabelEncoder()
y_encoded = le.fit_transform(y)

def safe_split(X_df, y_arr, test_ratio=0.2, random_state=42):
    n = len(y_arr)
    unique, counts = np.unique(y_arr, return_counts=True)
    min_count = counts.min() if len(counts) > 0 else 0
    # Dieu kien toi thieu de stratify on holdout: moi lop >= 2 mau, tong mau >= 5, it nhat 2 lop
    can_stratify = (len(unique) >= 2 and n >= 5 and min_count >= 2)
    if can_stratify:
        try:
            return train_test_split(
                X_df, y_arr, test_size=test_ratio, random_state=random_state, stratify=y_arr
            )
        except ValueError as e:
            print(f"Canh bao: stratify that bai ({e}). Thu chia khong stratify...")
    # Fallback: chia khong stratify, dam bao moi tap co it nhat 1 mau neu co the
    if n >= 2:
        test_size = max(1, int(round(test_ratio * n)))
        train_size = n - test_size
        if train_size < 1:
            train_size = 1
            test_size = n - 1
        rng = np.random.RandomState(random_state)
        perm = rng.permutation(n)
        test_idx = perm[:test_size]
        train_idx = perm[test_size:]
        return X_df.iloc[train_idx], X_df.iloc[test_idx], y_arr[train_idx], y_arr[test_idx]
    # Neu tap du lieu qua nho, khong the chia: dung toan bo de train, bo qua evaluate
    return X_df, None, y_arr, None

# Train/test split an toan
X_train, X_test, y_train, y_test = safe_split(X, y_encoded, test_ratio=0.2, random_state=42)

# Chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test_transformed = None if X_test is None else scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
if X_test is not None and y_test is not None:
    y_pred = model.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy (hold-out test): {acc*100:.2f}%")

    # Báo cáo chi tiết
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
else:
    train_acc = model.score(X_train, y_train)
    print(
        f"ℹ️ Du lieu qua nho/khong the chia tap test. Bao cao training accuracy de tham khao: {train_acc*100:.2f}%"
    )

# Save model + scaler + label encoder (standardized names)
MODEL_PATH = os.path.join(SAVE_DIR, "gesture_model.joblib")
SCALER_PATH = os.path.join(SAVE_DIR, "gesture_scaler.joblib")
ENCODER_PATH = os.path.join(SAVE_DIR, "gesture_label_encoder.joblib")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"💾 Đã lưu: {MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}")
