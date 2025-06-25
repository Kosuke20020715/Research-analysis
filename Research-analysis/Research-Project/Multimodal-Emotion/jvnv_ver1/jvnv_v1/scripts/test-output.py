import joblib
import numpy as np
import pandas as pd

output_csv = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\Test_Data\Output\output.csv"

# === モデル・エンコーダの読み込み ===
model = joblib.load(r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\models\lgb_model.pkl")
le = joblib.load(r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\models\label_encoder.pkl")

def has_meaningful_data(line):
    return any(cell.strip() for cell in line.strip().split(','))

# === 特徴量の読み取り ===
with open(output_csv, "r", encoding="utf-8") as f:
    lines = f.readlines()
    data_start = next(i for i, line in enumerate(lines) if "@data" in line.lower())

    # 最初に文字が書かれている（空白でない）行を抽出
    data_line = next(
    (line for line in lines[data_start + 1:] if has_meaningful_data(line)),
    None
)

vec = np.array([
    float(x) if x.strip() not in ["", "?"] else np.nan 
    for x in data_line.strip().split(",")[1:]
]).reshape(1, -1)

# === 予測とラベル逆変換 ===
y_pred_proba = model.predict(vec)
y_pred_index = np.argmax(y_pred_proba, axis=1)[0]
#labelとの逆変換
emotion_name = le.inverse_transform([y_pred_index])[0]

print(f"予測された感情: {emotion_name}")
