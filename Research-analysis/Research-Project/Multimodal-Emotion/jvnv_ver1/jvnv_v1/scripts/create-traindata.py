import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# === パス設定 ===
FEATURE_DIR = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\features"
OUTPUT_DIR = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# .csvファイルを収集
csv_files = glob(os.path.join(FEATURE_DIR, "*.csv"))

features = []
labels = []
filenames = []

for file_path in csv_files:
    filename = os.path.basename(file_path)
    base = os.path.splitext(filename)[0]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            data_start = next(i for i, line in enumerate(lines) if "@data" in line.lower()) + 2
            data_lines = lines[data_start:]
            data_line = next((line for line in data_lines if line.strip() != ""), None)

        vec = np.array([
            float(x) if x.strip() not in ["", "?"] else np.nan 
            for x in data_line.strip().split(",")[1:]
        ])
        label = base.split("_")[1] if len(base.split("_")) > 1 else "unknown"
        features.append(vec)
        labels.append(label)
        filenames.append(base)
    except Exception as e:
        print(f"[ERROR] {filename}: {e}")


X = np.array(features)
y = np.array(labels)

# train/testに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 保存（CSV）
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train
test_df = pd.DataFrame(X_test)
test_df['label'] = y_test

train_path = os.path.join(OUTPUT_DIR, "train_features.csv")
test_path = os.path.join(OUTPUT_DIR, "test_features.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Trainデータの先頭:")
print(train_df.head())
