import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    BertJapaneseTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
#read csv in Japanese
df = pd.read_csv("Results/wrime-ver2.csv", encoding="cp932")
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
df['readers_emotion_intensities'] = df.apply(
    lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1
)

# select only intense over 2
is_target = df['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df[is_target].copy()

# max emotion
df_wrime_target['label'] = df_wrime_target['readers_emotion_intensities'].map(lambda x: int(np.argmax(x)))

# text and label
texts = df_wrime_target['Sentence'].tolist()
labels = df_wrime_target['label'].tolist()
#train and test data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

#tokenize 
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese") #use pre-trained datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

#define datasets
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

#select model
model = BertForSequenceClassification.from_pretrained(
    "cl-tohoku/bert-base-japanese", num_labels=8
)
model.to(device)

#model settings
training_args = TrainingArguments(
    output_dir="./models/bert-emotion",
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy"
)

#evaluation 
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

#defeine Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

#train 
trainer.train()

#evaluate result
matrix = trainer.evaluate()
print(matrix)
