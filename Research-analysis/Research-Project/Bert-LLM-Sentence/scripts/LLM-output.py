import openai
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import ast

client = OpenAI(api_key='My_key')

# 感情分類（バッチ処理）
def classify_emotions_batch(texts):
    numbered_texts = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    prompt = f"""
以下の文章それぞれについて、0〜7の感情ラベルで分類してください。
出力は、各文に対するラベルをPythonのリスト形式 [0, 3, 2, ...] で返してください。

0: Joy  
1: Sadness  
2: Anticipation  
3: Surprise  
4: Anger  
5: Fear  
6: Disgust  
7: Trust

文章:
{numbered_texts}

回答のみ：[番号順のラベルリスト]
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        preds = ast.literal_eval(content)  # 安全にリスト形式を評価
        if isinstance(preds, list) and all(isinstance(x, int) for x in preds):
            return preds
    except Exception as e:
        print("Error during batch classification:", e)

    return [-1] * len(texts)

# 正解ラベル付きのサンプルを抽出
def preprocess_target(df):
    emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
    df['readers_emotion_intensities'] = df.apply(
        lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1
    )
    df_target = df[df['readers_emotion_intensities'].map(lambda x: max(x) >= 2)].copy()
    df_target['label'] = df_target['readers_emotion_intensities'].map(lambda x: int(np.argmax(x)))
    return df_target.reset_index(drop=True)

# 評価関数
def evaluate_predictions(preds, labels):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# --- メイン処理 ---

# CSVファイルの読み込み
df = pd.read_csv("Results/wrime-ver2.csv", encoding="cp932")

# 感情強度が一定以上のものだけを対象とする
df_target = preprocess_target(df)

# 処理件数（例：先頭100件のみ、全件処理したい場合は変更）
# N = 100
df_sample = df_target.copy()
texts = df_sample["Sentence"].tolist()
true_labels = df_sample["label"].tolist()

# バッチ処理によるGPT推論（10件ずつ）
gpt_preds = []
batch_size = 30

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    preds = classify_emotions_batch(batch)
    gpt_preds.extend(preds)
    print(f"[{i+1}~{i+len(batch)}] → {preds}")

# 結果DataFrame作成
df_result = pd.DataFrame({
    "text": texts,
    "gpt_emotion": gpt_preds,
    "true_label": true_labels
})

print("\n=== 結果一覧 ===")
print(df_result.head())

# 有効な予測のみで評価
valid_df = df_result[df_result["gpt_emotion"] != -1]
results = evaluate_predictions(valid_df["gpt_emotion"], valid_df["true_label"])

print("\n=== 評価指標 ===")
print(results)