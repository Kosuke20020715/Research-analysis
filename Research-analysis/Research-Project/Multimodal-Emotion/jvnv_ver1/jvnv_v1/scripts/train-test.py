import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import joblib

# === データの読み込み ===
train_df = pd.read_csv("C:/Users/redsw/Research/Research-analysis/Research-Project/Multimodal-Emotion/jvnv_ver1/train_features.csv")
test_df = pd.read_csv("C:/Users/redsw/Research/Research-analysis/Research-Project/Multimodal-Emotion/jvnv_ver1/test_features.csv")

# === 特徴量とラベルに分割 ===
X_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
X_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# === ラベルをエンコード（例: "anger" → 0） ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# === LightGBM Dataset に変換 ===
dtrain = lgb.Dataset(X_train, label=y_train_enc)
dvalid = lgb.Dataset(X_test, label=y_test_enc, reference=dtrain)

# === パラメータ設定 ===
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'verbosity': -1,
    'seed': 42
}

# === モデル学習 ===
model = lgb.train(
    params,
    dtrain,
    valid_sets=[dvalid],
    valid_names=["valid"],
    num_boost_round=100
)

# === 予測 ===
y_pred_proba = model.predict(X_test)
y_pred_enc = np.argmax(y_pred_proba, axis=1)

# === 逆変換（数値 → 感情ラベル） ===
y_pred_labels = le.inverse_transform(y_pred_enc)
y_true_labels = le.inverse_transform(y_test_enc)

# === 評価 ===
print("\n[Accuracy]")
print(accuracy_score(y_true_labels, y_pred_labels))

print("\n[分類レポート]")
print(classification_report(y_true_labels, y_pred_labels))

# === 混同行列 ===
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#モデルパス
SAVE_PATH = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\models"
MODEL_PATH =  SAVE_PATH + "\lgb_model.pkl"
ENCODER_PATH = SAVE_PATH + "\label_encoder.pkl"

# モデルとラベルエンコーダを保存
joblib.dump(model, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)