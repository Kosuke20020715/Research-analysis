import whisper
import os
import pandas as pd
import glob
from pathlib import Path

model = whisper.load_model("small") 

# ベースディレクトリのパス（あなたの環境に合わせて修正）
BASE_DIR = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\jvnv_v1"

# 出力用
results = []

emotions = ["anger", "disgust", "fear", "happy", "sad", "surprise"]

for speaker_dir in os.listdir(BASE_DIR):  # F1, F2, M1, M2...
    speaker_path = os.path.join(BASE_DIR, speaker_dir)
    if not os.path.isdir(speaker_path):
        continue

    for emotion in emotions:
        emotion_path = os.path.join(speaker_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        # 再帰的に.wav収集
        wav_files = glob.glob(os.path.join(emotion_path, "**", "*.wav"), recursive=True)

        for wav_path in wav_files:
            try:
                filename = Path(wav_path).name
                result = model.transcribe(wav_path)
                results.append({
                    "speaker": speaker_dir,
                    "filename": filename,
                    "text": result["text"],
                    "emotion": emotion
                })
            except Exception as e:
                print(f"[ERROR] {wav_path}: {e}")

# CSVとして保存
output_csv_path = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Bert-LLM-Sentence\Results\emotion_transcriptions.csv"
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False, encoding="utf-8")