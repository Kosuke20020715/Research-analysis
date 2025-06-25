import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertJapaneseTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# CSVの読み込み
df = pd.read_csv(r"C:\Users\redsw\Research\Research-analysis\Research-Project\Bert-LLM-Sentence\Results\emotion_transcriptions.csv", encoding="cp932")

# テキストと感情ラベルを抽出
texts = df["text"].tolist()
labels = df["emotion"].tolist()

# ラベルを数値に変換
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# ラベル一覧の保存（逆変換のため）
label_list = list(label_encoder.classes_)
print("感情ラベル一覧:", label_list)

# データを学習/検証に分割
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
)

# トークナイザー
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
# トークナイズ
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

#datasetの定義
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

num_labels = 6

# モデルのロード
model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese",
    num_labels=num_labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir="./models/bert-emotion",          # モデルの保存先
    learning_rate=1e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",                 # 各エポックごとに評価
    save_strategy="epoch",                       # 各エポックごとに保存
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 学習の実行
trainer.train()

eval_result = trainer.evaluate()
print(eval_result)

# モデルの保存
model.save_pretrained("./models/bert-emotion")
tokenizer.save_pretrained("./models/bert-emotion")
# label保存
with open("./models/bert-emotion/emotion_labels.json", "w", encoding="utf-8") as f:
    json.dump(label_list, f, ensure_ascii=False)


# # 入力をモデルに渡す（ここでは最初の1文のみ推論）
# input_ids = train_encodings["input_ids"][:1].to(device)
# attention_mask = train_encodings["attention_mask"][:1].to(device)

# with torch.no_grad():
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     logits = outputs.logits
#     predicted_label = torch.argmax(logits, dim=1).item()

# print(predicted_label)

