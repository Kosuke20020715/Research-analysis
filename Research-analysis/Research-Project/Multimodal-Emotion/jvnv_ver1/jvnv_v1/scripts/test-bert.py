from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import json 
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとトークナイザの読み込み
model = BertForSequenceClassification.from_pretrained("./models/bert-emotion")
tokenizer = BertJapaneseTokenizer.from_pretrained("./models/bert-emotion")
model.to(device)

# 感情名のロード
with open("./models/bert-emotion/emotion_labels.json", "r", encoding="utf-8") as f:
    label_list = json.load(f)

# 推論
inputs = tokenizer("そんなことある？", return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
predicted_label = outputs.logits.argmax(dim=1).item()

# 感情名に変換
predicted_emotion = label_list[predicted_label]
print(f"予測された感情: {predicted_emotion}")

