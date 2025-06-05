import whisper
import os
import pandas as pd
import glob
from pathlib import Path

model = whisper.load_model("small") 
results = []

#0:normal/1:angry/2:happy
label_dirs = {
    0: "data/tsuchiya_normal/tsuchiya_normal/*.wav",
    1: "data/tsuchiya_angry/tsuchiya_angry/*.wav",
    2: "data/tsuchiya_happy/tsuchiya_happy/*.wav"
}

for label, path_glob in label_dirs.items():
    files = glob.glob(path_glob) #全ファイル
    for i, filepath in enumerate(files):
        filename = Path(filepath).name
        result = model.transcribe(filepath)
        results.append({
            "filename": filename,
            "label": label,
            "text": result["text"]
        })
        
#csv出力
df = pd.DataFrame(results)
df.to_csv("Results/transcriptions.csv", index=False, encoding="utf-8")