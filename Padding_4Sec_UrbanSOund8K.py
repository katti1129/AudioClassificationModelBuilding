import os
import librosa
import soundfile as sf

# 入力フォルダと出力フォルダ
input_dir = r"C:\Users\cpsla\PycharmProjects\lab_lstm\ModelBuilding\data\UrbanSound8K_split_safe"  # UrbanSound8Kのaudioフォルダ（例：fold1〜fold10）
output_dir = r"C:\Users\cpsla\PycharmProjects\lab_lstm\ModelBuilding\data\UrbanSound8K_split_4sec"  # パディング後の保存先

# サンプリングレートと目標長さ
TARGET_SR = 44100
TARGET_LENGTH_SEC = 4
TARGET_LENGTH_SAMPLES = TARGET_SR * TARGET_LENGTH_SEC


def pad_or_trim_audio(y, target_length):
    current_length = len(y)
    if current_length < target_length:
        padding = target_length - current_length
        y_padded = librosa.util.fix_length(y, size=target_length)
        return y_padded
    else:
        return y[:target_length]


# 全ファイル処理
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".wav"):
            input_path = os.path.join(root, file)
            y, sr = librosa.load(input_path, sr=TARGET_SR)

            y_fixed = pad_or_trim_audio(y, TARGET_LENGTH_SAMPLES)

            # 出力先のフォルダを保持
            rel_path = os.path.relpath(root, input_dir)
            save_folder = os.path.join(output_dir, rel_path)
            os.makedirs(save_folder, exist_ok=True)
            output_path = os.path.join(save_folder, file)

            sf.write(output_path, y_fixed, TARGET_SR)

print("全ファイルが4秒（176400サンプル）に統一されました。")
