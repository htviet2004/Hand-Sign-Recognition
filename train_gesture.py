import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
# Thêm import cho vẽ biểu đồ
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, log_loss

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

    # === VẼ CONFUSION MATRIX ===
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(cmap="Blues", xticks_rotation=45, colorbar=False)
        plt.title("Confusion Matrix (Test)")
        plt.tight_layout()
        cm_path = os.path.join(SAVE_DIR, "cm_test.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"📈 Đã lưu biểu đồ: {cm_path}")
    except Exception as e:
        print(f"⚠️ Không thể vẽ confusion matrix: {e}")

else:
    train_acc = model.score(X_train, y_train)
    print(
        f"ℹ️ Du lieu qua nho/khong the chia tap test. Bao cao training accuracy de tham khao: {train_acc*100:.2f}%"
    )

# === LEARNING CURVE (train/val accuracy theo kích thước tập) ===
try:
    # Số mẫu tối thiểu mỗi lớp để KFold stratify an toàn
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_count = int(counts.min()) if len(counts) > 0 else 0
    n_splits = max(2, min(5, min_count))  # ít nhất 2, tối đa 5
    if n_splits >= 2 and len(unique) >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        estimator = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        )
        train_sizes, train_scores, val_scores = learning_curve(
            estimator,
            X.values, y_encoded,
            cv=cv, scoring="accuracy", n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 5),
            shuffle=True
        )
        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        plt.figure(figsize=(7,5))
        plt.plot(train_sizes, train_mean, "o-", label="Train acc")
        plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
        plt.plot(train_sizes, val_mean, "o-", label="CV acc")
        plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
        plt.xlabel("Số mẫu dùng để train")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve (RandomForest)")
        plt.legend()
        plt.grid(alpha=0.3)
        lc_path = os.path.join(SAVE_DIR, "learning_curve.png")
        plt.tight_layout()
        plt.savefig(lc_path, dpi=150)
        plt.close()
        print(f"📈 Đã lưu biểu đồ: {lc_path}")
    else:
        print("ℹ️ Bỏ qua learning curve (không đủ dữ liệu hoặc chỉ 1 lớp).")
except Exception as e:
    print(f"⚠️ Không thể vẽ learning curve: {e}")

# === OOB / TEST CURVE THEO SỐ CÂY (n_estimators) ===
try:
    # Chỉ chạy khi có tối thiểu vài chục mẫu để OOB có ý nghĩa
    if X_test is not None and y_test is not None and len(y_train) >= 20:
        trees_list = list(range(10, 301, 10))  # 10 → 300 cây
        # Dùng dữ liệu đã chuẩn hóa cho RF
        Xtr = X_train
        Xte = X_test_transformed
        rf = RandomForestClassifier(
            n_estimators=10, warm_start=True, oob_score=True,
            class_weight="balanced", random_state=42
        )
        oob_acc, test_acc, test_logloss = [], [], []
        for n_trees in trees_list:
            rf.set_params(n_estimators=n_trees)
            rf.fit(Xtr, y_train)
            # OOB accuracy
            oob_acc.append(getattr(rf, "oob_score_", np.nan))
            # Test accuracy và log-loss
            proba = rf.predict_proba(Xte)
            yp = np.argmax(proba, axis=1)
            test_acc.append(accuracy_score(y_test, yp))
            try:
                test_logloss.append(log_loss(y_test, proba, labels=range(len(le.classes_))))
            except Exception:
                test_logloss.append(np.nan)

        fig, ax1 = plt.subplots(figsize=(7,5))
        ax1.plot(trees_list, test_acc, "o-", color="tab:blue", label="Test acc")
        ax1.plot(trees_list, oob_acc,  "o--", color="tab:green", label="OOB acc")
        ax1.set_xlabel("Số cây (n_estimators)")
        ax1.set_ylabel("Accuracy")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(trees_list, test_logloss, "s-", color="tab:red", label="Test log-loss")
        ax2.set_ylabel("Log-loss")
        # Gộp legend
        lines, labels = [], []
        for ax in (ax1, ax2):
            l, lab = ax.get_legend_handles_labels()
            lines += l; labels += lab
        ax1.legend(lines, labels, loc="best")
        plt.title("Hiệu năng theo số cây (RF)")
        oob_path = os.path.join(SAVE_DIR, "oob_test_curves.png")
        plt.tight_layout()
        plt.savefig(oob_path, dpi=150)
        plt.close()
        print(f"📈 Đã lưu biểu đồ: {oob_path}")
    else:
        print("ℹ️ Bỏ qua OOB/test curves (không có test set hoặc dữ liệu quá ít).")
except Exception as e:
    print(f"⚠️ Không thể vẽ OOB/test curves: {e}")

# Save model + scaler + label encoder (standardized names)
MODEL_PATH = os.path.join(SAVE_DIR, "gesture_model.joblib")
SCALER_PATH = os.path.join(SAVE_DIR, "gesture_scaler.joblib")
ENCODER_PATH = os.path.join(SAVE_DIR, "gesture_label_encoder.joblib")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"💾 Đã lưu: {MODEL_PATH}, {SCALER_PATH}, {ENCODER_PATH}")
