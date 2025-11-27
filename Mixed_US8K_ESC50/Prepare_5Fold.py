# save as: build_dataset_full_pipeline.py

import librosa
import numpy as np
import shutil
import random
import soundfile as sf
from pathlib import Path
import pandas as pd

# =========================================================
# 0) 設定
# =========================================================
US8K = Path("../../data/UrbanSound8K")
ESC50 = Path("../../data/ESC-50")
OUT = Path("../../data/combined_wav_16k_mono")
BALANCED = OUT / "balanced_400"
FOLDS_DIR = OUT / "folds"
WINDOWED_DIR = OUT / "windowed"

OUT.mkdir(exist_ok=True)
BALANCED.mkdir(exist_ok=True)
FOLDS_DIR.mkdir(exist_ok=True)
WINDOWED_DIR.mkdir(exist_ok=True)

TARGET_SR = 16000
WINDOW = 1.5
HOP = 0.5
SAMPLES = int(TARGET_SR * WINDOW)
HOP_SAMPLES = int(TARGET_SR * HOP)

TARGET_CLASSES = ["siren", "other", "silence"]


# =========================================================
# 1) データ抽出（UrbanSound8K → siren, ESC → other）
# =========================================================

def extract_and_convert(src, dst):
    """ 16kHz + mono に変換して保存 """
    audio, _ = librosa.load(src, sr=TARGET_SR, mono=True)
    sf.write(dst, audio, TARGET_SR)


def step1_extract():
    print("=== Step1: データ抽出 + 16kHz / mono 統一 ===")

    us_meta = pd.read_csv(US8K / "metadata/UrbanSound8K.csv")
    esc_meta = pd.read_csv(ESC50 / "meta/esc50.csv")

    # UrbanSound8K → siren のみ
    for _, r in us_meta.iterrows():
        if r["class"] != "siren":
            continue

        out_dir = OUT / "siren"
        out_dir.mkdir(exist_ok=True)

        src = US8K / "audio" / f"fold{r['fold']}" / r["slice_file_name"]
        dst = out_dir / r["slice_file_name"]
        extract_and_convert(src, dst)

    # ESC-50 → other（speech, scream, wind）
    ESC_MAP = {"speech": "other", "scream": "other", "wind": "other"}

    for _, r in esc_meta.iterrows():
        if r["category"] not in ESC_MAP:
            continue

        out_dir = OUT / "other"
        out_dir.mkdir(exist_ok=True)

        src = ESC50 / "audio" / r["filename"]
        dst = out_dir / r["filename"]
        extract_and_convert(src, dst)

    # silence
    sil_dir = OUT / "silence"
    sil_dir.mkdir(exist_ok=True)

    for i in range(400):  # silence も400枚
        silence = np.zeros(SAMPLES, dtype=np.float32)
        sf.write(sil_dir / f"silence_{i}.wav", silence, TARGET_SR)

    print("✔ Step1 完了")


# =========================================================
# 2) クラスごとに400枚に均一化
# =========================================================

def step2_balance():
    print("=== Step2: クラス枚数を400に統一 ===")

    for cls in TARGET_CLASSES:
        src_dir = OUT / cls
        dst_dir = BALANCED / cls
        dst_dir.mkdir(exist_ok=True)

        files = list(src_dir.glob("*.wav"))

        if len(files) > 400:
            files = random.sample(files, 400)
        else:
            while len(files) < 400:
                files.append(random.choice(files))

        for i, f in enumerate(files):
            shutil.copyfile(f, dst_dir / f"{cls}_{i}.wav")

    print("✔ Step2 完了")


# =========================================================
# 3) Fold 分割（5fold or 9fold）
# =========================================================

def step3_make_folds(k=5):
    print(f"=== Step3: {k}-Fold 分割 ===")

    for fold in range(k):
        fold_dir = FOLDS_DIR / f"fold{fold+1}"
        fold_dir.mkdir(exist_ok=True)

        for cls in TARGET_CLASSES:
            (fold_dir / cls).mkdir(exist_ok=True)

    for cls in TARGET_CLASSES:
        files = sorted((BALANCED / cls).glob("*.wav"))
        random.shuffle(files)

        fold_size = len(files) // k

        for i in range(k):
            fold_files = files[i * fold_size:(i + 1) * fold_size]

            for f in fold_files:
                shutil.copyfile(f, FOLDS_DIR / f"fold{i+1}" / cls / f.name)

    print("✔ Step3 完了")


# =========================================================
# 4) スライドウィンドウ適用 (1.5s, hop 0.5s)
# =========================================================

def step4_sliding_window(k=5):
    print("=== Step4: スライドウィンドウ生成 ===")

    for fold in range(1, k + 1):
        src_dir = FOLDS_DIR / f"fold{fold}"
        dst_dir = WINDOWED_DIR / f"fold{fold}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        for cls in TARGET_CLASSES:
            out_cls_dir = dst_dir / cls
            out_cls_dir.mkdir(exist_ok=True)

            for file in (src_dir / cls).glob("*.wav"):
                audio, _ = librosa.load(file, sr=TARGET_SR)

                idx = 0
                start = 0
                while start + SAMPLES <= len(audio):
                    chunk = audio[start:start + SAMPLES]
                    sf.write(out_cls_dir / f"{file.stem}_{idx}.wav", chunk, TARGET_SR)
                    idx += 1
                    start += HOP_SAMPLES

    print("✔ Step4 完了")


# =========================================================
# 実行フロー
# =========================================================

if __name__ == "__main__":
    step1_extract()           # データ抽出 + 16kHz mono
    step2_balance()           # 400枚均一化
    step3_make_folds(k=5)     # 5fold / 9fold に変更も可
    step4_sliding_window(k=5) # スライドウィンドウ生成
    print("\n=== 全工程 完了 ===")
