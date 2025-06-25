import os
import subprocess
from glob import glob
from tqdm import tqdm

# === ユーザー設定 ===
SMILExtract_PATH = r"C:\Users\redsw\opensmile\build\progsrc\smilextract\Release\SMILExtract.exe"
CONFIG_PATH = r"C:\Users\redsw\opensmile\config\compare16\ComParE_2016.conf"
INPUT_DIR = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\jvnv_v1"   # .wavが入っているディレクトリ
OUTPUT_DIR = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\features"  # .csv出力先

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_paths = glob(os.path.join(INPUT_DIR, "**", "*.wav"), recursive=True)
    if not wav_paths:
        print(f"[ERROR] WAV ファイルが見つかりませんでした: {INPUT_DIR}")
        return

    print(f"[INFO] 合計 {len(wav_paths)} ファイルを処理します。\n")

    for wav in tqdm(wav_paths, desc="Extracting"):
        # 元ファイル名（拡張子なし）を取得
        basename = os.path.splitext(os.path.basename(wav))[0]
        # 出力パス
        out_csv  = os.path.join(OUTPUT_DIR, f"{basename}.csv")
        
        # openSMILE 実行コマンド
        cmd = [
            SMILExtract_PATH,
            "-C", CONFIG_PATH,
            "-I", wav,
            "-O", out_csv
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {basename} の抽出に失敗: {e.stderr.decode().strip()}")
            continue

    print("\n[INFO] 特徴抽出が完了しました。")
    print(f"[INFO] 出力先フォルダ: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()