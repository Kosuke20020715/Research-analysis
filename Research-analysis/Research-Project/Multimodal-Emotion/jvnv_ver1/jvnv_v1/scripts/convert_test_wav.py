import subprocess
import os

# パスの設定
opensmile_exe = r"C:\Users\redsw\opensmile\build\progsrc\smilextract\Release\SMILExtract.exe"
config_path = r"C:\Users\redsw\opensmile\config\is09-13\IS13_ComParE.conf"
input_wav = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\Test_Data\Input\tsuchiya_happy_001.wav"
output_csv = r"C:\Users\redsw\Research\Research-analysis\Research-Project\Multimodal-Emotion\jvnv_ver1\Test_Data\Output\output.csv"

# output.csv が存在する場合は削除
if os.path.exists(output_csv):
    os.remove(output_csv)

# OpenSMILEを実行
cmd = [
    opensmile_exe,
    "-C", config_path,
    "-I", input_wav,
    "-O", output_csv
]

result = subprocess.run(cmd, capture_output=True, text=True)

# エラーハンドリング
if result.returncode == 0:
    print("音声特徴量の抽出が完了しました。")
else:
    print("エラーが発生しました:")
    print(result.stderr)
