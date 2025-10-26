import soundfile as sf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model# type: ignore
from pathlib import Path
import time
import librosa

# --- 相対パス設定 ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "../code/model/best_resnet_crnn_7_model.keras"
DATA_PATH = BASE_DIR / "../data/1-Ambulance/Ambulance-Siren01-1.wav"

# --- モデル読み込み ---
model = load_model(MODEL_PATH)

# --- 音源読み込み ---
waveform, sr = sf.read(DATA_PATH)
if sr != 16000:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    sr = 16000

# --- ストリーミング再現 ---
CHUNK_SIZE = 16000  # 1秒分
for i in range(0, len(waveform), CHUNK_SIZE):
    chunk = waveform[i:i+CHUNK_SIZE]
    if len(chunk) < CHUNK_SIZE:
        break

    # ---- 特徴量生成（学習時と同じ形にする）----
    mel_spec = librosa.feature.melspectrogram(
        y=chunk, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # [128, 時間フレーム数] → [128, 47] に固定
    # 時間方向を47フレームにリサイズ（またはパディング/切り出し）
    if mel_spec_db.shape[1] < 47:
        pad_width = 47 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > 47:
        mel_spec_db = mel_spec_db[:, :47]

    # --- (128, 47, 3) に拡張 ---
    mel_spec_db = np.stack([mel_spec_db]*3, axis=-1)  # RGB風に3チャネル化
    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # (1, 128, 47, 3)

    # ---- 推論 ----
    pred = model.predict(mel_spec_db)
    top_class = np.argmax(pred)
    conf = np.max(pred)
    print(f"chunk {i//CHUNK_SIZE}: class={top_class}, confidence={conf:.2f}")

    time.sleep(1.0)