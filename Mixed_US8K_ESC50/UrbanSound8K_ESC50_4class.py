import pandas as pd
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa

# === パス設定 ===
US8K = Path("../../data/UrbanSound8K")
ESC50 = Path("../../data/ESC-50")
OUT = Path("../../data/combined_wav_16k_mono")  # ← 出力フォルダ
OUT.mkdir(exist_ok=True)

# === メタ読み込み ===
us_meta = pd.read_csv(US8K / "metadata/UrbanSound8K.csv")
esc_meta = pd.read_csv(ESC50 / "meta/esc50.csv")

# === 基本設定 ===
SR = 16000        # リサンプリング先のサンプリングレート
DUR = 3.0         # 無音の秒数
SAMPLES = int(SR * DUR)

# ---- 16kHz + mono に統一して保存する関数 ----
def convert_and_save(src, dst):
    audio, _ = librosa.load(src, sr=SR, mono=True)
    sf.write(dst, audio, SR)


# =========================================================
# ① UrbanSound8K → siren を 16kHz-mono にして保存
# =========================================================

TARGET_US8K = {"siren": "siren"}  # car_horn は今回は使わない

for _, r in us_meta.iterrows():
    cls = r["class"]

    if cls not in TARGET_US8K:
        continue

    out_dir = OUT / TARGET_US8K[cls]
    out_dir.mkdir(exist_ok=True)

    src = US8K / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
    dst = out_dir / r["slice_file_name"]

    convert_and_save(src, dst)


# =========================================================
# ② ESC-50 → other（speech / scream / wind）を保存
# =========================================================

ESC_TO_FINAL = {
    "scream": "other",
    "speech": "other",
    "wind": "other",
}

for _, r in esc_meta.iterrows():
    cat = r["category"]

    if cat not in ESC_TO_FINAL:
        continue

    final_cls = ESC_TO_FINAL[cat]
    out_dir = OUT / final_cls
    out_dir.mkdir(exist_ok=True)

    src = ESC50 / "audio" / r["filename"]
    dst = out_dir / r["filename"]

    convert_and_save(src, dst)


# =========================================================
# ③ silence (1.5秒) を 16kHz-mono のまま保存
# =========================================================

silence_dir = OUT / "silence"
silence_dir.mkdir(exist_ok=True)

for i in range(50):
    silence = np.zeros(SAMPLES, dtype=np.float32)
    sf.write(silence_dir / f"silence_{i}.wav", silence, SR)

print("✔ 16kHz + mono 統一 wav データセット生成 完了")
