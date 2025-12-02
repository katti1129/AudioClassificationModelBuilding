# -*- coding: utf-8 -*-
# save as: Step1_LeakyFree_Split.py
#UrbanSound8K（siren）・ESC-50（other）を リークフリーに5-foldへ分割。silenceを生成．

import pandas as pd
from pathlib import Path
import soundfile as sf
import numpy as np
import librosa
import shutil
from tqdm import tqdm

# ================= 設定 =================
SR = 16000
US8K_DIR = Path("../../data/UrbanSound8K")
ESC50_DIR = Path("../../data/ESC-50")
OUT_ROOT = Path("../../data/clean_folds_5fold")  # 新しい出力先

# --- US8Kの設定 ---
TARGET_US8K = {"siren": "siren"}

# --- ESC-50の設定 (修正版: 実在するカテゴリ名に変更) ---
# これで合計 20カテゴリ × 40枚 = 800枚 の other が確保されます
TARGET_ESC50 = {
    # --- 自然音 ---
    "wind": "other",
    "rain": "other",
    "thunderstorm": "other",
    "sea_waves": "other",
    "chirping_birds": "other",

    # --- 人間・動物 ---
    "crying_baby": "other",
    "laughing": "other",
    "coughing": "other",
    "footsteps": "other",
    "dog": "other",
    "cat": "other",
    "crow": "other",

    # --- 乗り物・機械 ---
    "engine": "other",
    "train": "other",
    "airplane": "other",
    "helicopter": "other",
    "car_horn": "other",  # 難例として重要
    "vacuum_cleaner": "other",

    # --- 衝撃音・警報音（Sirenとの区別用） ---
    "glass_breaking": "other",
    "clock_alarm": "other"
}

# --- Silenceの設定 ---
NUM_SILENCE_PER_FOLD = 100  # otherが増えたので少し増量
SILENCE_DUR = 3.0  # 生成する長さ(秒)


# ========================================

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def convert_and_save(src_path, dst_path):
    """16kHzモノラルに変換して保存"""
    try:
        y, _ = librosa.load(src_path, sr=SR, mono=True)
        sf.write(dst_path, y, SR)
    except Exception as e:
        print(f"[Error] Failed to load {src_path}: {e}")


def get_dest_fold_us8k(orig_fold):
    """
    US8K (1-10) を 5-fold (1-5) にマッピング
    fold1,2 -> 1
    fold3,4 -> 2
    ...
    """
    return (orig_fold - 1) // 2 + 1


def generate_noise(n_samples, mode="white"):
    if mode == "white":
        x = np.random.randn(n_samples)
    elif mode == "pink":
        # 簡易ピンクノイズ (1/f)
        uneven = n_samples % 2
        X = np.random.randn(n_samples // 2 + 1 + uneven) + 1j * np.random.randn(n_samples // 2 + 1 + uneven)
        S = np.sqrt(1.0 / np.arange(1, n_samples // 2 + 2 + uneven))
        x = np.fft.irfft(X * S).real
        if uneven: x = x[:-1]
    else:
        x = np.zeros(n_samples)

    # 正規化 (-30dB ~ -25dB 程度にスケール)
    x = x.astype(np.float32)
    max_val = np.max(np.abs(x)) + 1e-8
    x = x / max_val
    scale = 10 ** (np.random.uniform(-30, -25) / 20.0)
    return x * scale


def main():
    if OUT_ROOT.exists():
        print(f"[Warning] {OUT_ROOT} は既に存在します．削除して作り直しますか？ (y/n)")
        ans = input().strip().lower()
        if ans == 'y':
            shutil.rmtree(OUT_ROOT)
        else:
            print("中止しました．")
            return

    # --- 1. UrbanSound8K の処理 ---
    print("=== Processing UrbanSound8K (Siren) ===")
    meta_us = pd.read_csv(US8K_DIR / "metadata/UrbanSound8K.csv")

    for _, row in tqdm(meta_us.iterrows(), total=len(meta_us)):
        cls = row["class"]
        if cls not in TARGET_US8K:
            continue

        orig_fold = row["fold"]
        dest_fold = get_dest_fold_us8k(orig_fold)
        final_cls = TARGET_US8K[cls]

        # 保存先: foldX/siren/filename
        dst_dir = OUT_ROOT / f"fold{dest_fold}" / final_cls
        ensure_dir(dst_dir)

        src_file = US8K_DIR / "audio" / f"fold{orig_fold}" / row["slice_file_name"]
        dst_file = dst_dir / row["slice_file_name"]

        convert_and_save(src_file, dst_file)

    # --- 2. ESC-50 の処理 ---
    print("\n=== Processing ESC-50 (Other) ===")
    meta_esc = pd.read_csv(ESC50_DIR / "meta/esc50.csv")

    for _, row in tqdm(meta_esc.iterrows(), total=len(meta_esc)):
        cat = row["category"]
        if cat not in TARGET_ESC50:
            continue

        orig_fold = row["fold"]  # 1-5
        dest_fold = orig_fold  # ESC-50はそのまま使う
        final_cls = TARGET_ESC50[cat]

        dst_dir = OUT_ROOT / f"fold{dest_fold}" / final_cls
        ensure_dir(dst_dir)

        src_file = ESC50_DIR / "audio" / row["filename"]
        dst_file = dst_dir / row["filename"]

        convert_and_save(src_file, dst_file)

    # --- 3. Silence (Noise) 生成 ---
    print("\n=== Generating Silence (White/Pink/Zero) ===")
    samples = int(SR * SILENCE_DUR)

    for f in range(1, 6):  # fold 1 to 5
        dst_dir = OUT_ROOT / f"fold{f}" / "silence"
        ensure_dir(dst_dir)

        for i in range(NUM_SILENCE_PER_FOLD):
            # 3種類をランダムに
            mode = np.random.choice(["white", "pink", "zero"], p=[0.4, 0.4, 0.2])
            y = generate_noise(samples, mode)

            fname = f"silence_gen_{mode}_{i}.wav"
            sf.write(dst_dir / fname, y, SR)

    print("\n[Done] リークフリーなデータセット分割が完了しました．")
    print(f"保存先: {OUT_ROOT}")


if __name__ == "__main__":
    main()