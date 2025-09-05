import librosa
import soundfile as sf
import os
from pathlib import Path


def trim_silence_from_files(input_folder, output_folder, top_db=20):
    """
    指定されたフォルダ内の全てのWAVファイルの先頭と末尾の無音をカットする．

    Args:
        input_folder (str): 入力WAVファイルが含まれるフォルダのパス．
        output_folder (str): 出力先フォルダのパス．
        top_db (int): 無音と判断するしきい値（dB）．この値より小さい部分が無音と見なされる．
                      値が小さいほど、より静かな部分まで音声として残す．
    """
    # 出力先フォルダが存在しない場合は作成
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # フォルダ内のファイルをループ処理
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 音声ファイルを読み込み
                y, sr = librosa.load(input_path, sr=None)

                # 無音部分をカット
                y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

                # カットした音声を保存
                sf.write(output_path, y_trimmed, sr)

                original_duration = len(y) / sr
                trimmed_duration = len(y_trimmed) / sr
                print(f"処理完了: {filename} ({original_duration:.2f}s -> {trimmed_duration:.2f}s)")

            except Exception as e:
                print(f"エラー: {filename} の処理中に問題が発生しました - {e}")


# --- 実行 ---
if __name__ == "__main__":
    # ESC-50データセットが含まれるフォルダを想定
    # このパスはご自身の環境に合わせて変更してください
    INPUT_AUDIO_FOLDER = "./data/ESC-50-master/audio/"
    OUTPUT_TRIMMED_FOLDER = "./data/ESC-50-master/audio_trimmed/"

    # 無音除去処理を実行
    trim_silence_from_files(INPUT_AUDIO_FOLDER, OUTPUT_TRIMMED_FOLDER)