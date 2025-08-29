import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

np.random.seed(42)

# ==== LOAD DATA ====
df = pd.read_csv("data/my_gesture_landmarks.csv")

# Kiểm tra và loại bỏ dòng thiếu dữ liệu
df = df.dropna()
print(f"Số mẫu sau khi loại bỏ thiếu dữ liệu: {len(df)}")

X = df.drop("gesture_label", axis=1)
y = df["gesture_label"]

# Encode label thành số
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split với stratify để giữ tỷ lệ lớp
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")

# Báo cáo chi tiết
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model + scaler + label encoder
joblib.dump(model, "data/gesture_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")
joblib.dump(le, "data/label_encoder.pkl")
print("💾 Đã lưu model, scaler và label encoder.")
