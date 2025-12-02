#枚数を数えて400マイに削減．silenceを追加．

import random
import shutil
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# ===== 設定 =====
SR = 16000
TARGET = 400  # siren / other / silence のターゲット枚数

OUT = Path("../../data/combined_wav_16k_mono")   # 4クラスが入っているフォルダ
ESC50 = Path("../../data/ESC-50")       # ESC-50 のルート (wind 用)


# ===== ユーティリティ =====
def list_wavs(cls: str):
    d = OUT / cls
    d.mkdir(exist_ok=True)
    return sorted(d.glob("*.wav"))


def count_all():
    print("=== 現在の枚数 ===")
    for cls in ["siren", "other", "silence"]:
        files = list_wavs(cls)
        print(f"{cls:8s}: {len(files)} files")
    print("================")


def downsample_to(files, target, unused_dir: Path):
    """target 枚だけ残し、残りを unused_dir に移動"""
    n = len(files)
    if n <= target:
        print(f"  {n} 枚 <= {target} 枚なのでダウンサンプリングなし")
        return

    print(f"  {n} 枚 -> {target} 枚にダウンサンプリングします")
    files = list(files)
    random.shuffle(files)
    keep = files[:target]
    drop = files[target:]

    unused_dir.mkdir(exist_ok=True)
    for f in drop:
        f.rename(unused_dir / f.name)


def oversample_to(files, target):
    """ランダム複製して target 枚まで増やす"""
    n = len(files)
    if n == 0:
        print("  元データが 0 枚なのでオーバーサンプリングできません")
        return
    if n >= target:
        print(f"  {n} 枚 >= {TARGET} 枚なのでオーバーサンプリングなし")
        return

    print(f"  {n} 枚 -> {target} 枚にオーバーサンプリングします")
    files = list(files)
    cur_files = list(files)
    i = 0
    while len(cur_files) < target:
        src = random.choice(files)
        new_name = src.with_name(src.stem + f"_dup{i}.wav")
        shutil.copyfile(src, new_name)
        cur_files.append(new_name)
        i += 1


def scale_to_db(x: np.ndarray, db_range=(-30, -25)):
    """信号 x を指定した dBFS 範囲でスケーリング"""
    x = x.astype(np.float32)
    max_abs = np.max(np.abs(x)) + 1e-8
    x = x / max_abs
    db = random.uniform(*db_range)
    scale = 10 ** (db / 20.0)
    return x * scale


def generate_white_noise(n_samples: int):
    x = np.random.randn(n_samples).astype(np.float32)
    x = x / (np.max(np.abs(x)) + 1e-8)
    return x


def generate_pink_noise(n_samples: int):
    """
    簡易 Pink Noise 生成 (FFT ベース)
    正確さよりもデータ拡張としての用途を優先
    """
    uneven = n_samples % 2
    X = np.random.randn(n_samples // 2 + 1 + uneven) + 1j * np.random.randn(
        n_samples // 2 + 1 + uneven
    )
    S = np.sqrt(1.0 / np.arange(1, n_samples // 2 + 2 + uneven))
    y = np.fft.irfft(X * S).real
    if uneven:
        y = y[:-1]
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-8)
    return y[:n_samples]


def load_wind_file_list():
    """ESC-50 の wind クラスのファイル一覧を返す"""
    meta_path = ESC50 / "meta/esc50.csv"
    if not meta_path.exists():
        print("  ESC-50 のメタファイルが見つかりませんでした（wind ノイズはスキップ）")
        return []

    meta = pd.read_csv(meta_path)
    wind_rows = meta[meta["category"] == "wind"]
    files = []
    for _, r in wind_rows.iterrows():
        path = ESC50 / "audio" / r["filename"]
        if path.exists():
            files.append(path)
    if not files:
        print("  ESC-50 の wind クラスの音声が見つかりません（wind ノイズはスキップ）")
    else:
        print(f"  ESC-50 wind ファイル {len(files)} 件を使用します")
    return files


def random_wind_clip(wind_files, length=SR):
    """wind ファイルからランダムに 1 秒分を切り出し"""
    src = random.choice(wind_files)
    y, _ = librosa.load(src, sr=SR, mono=True)
    if len(y) < length:
        # 足りなければループさせる
        repeat = length // len(y) + 1
        y = np.tile(y, repeat)
    if len(y) > length:
        start = random.randint(0, len(y) - length)
        y = y[start : start + length]
    return y[:length].astype(np.float32)


def augment_silence_to_target(target=TARGET):
    """silence クラスを Pink / White / Wind / Pure で target 枚まで増やす"""
    silence_dir = OUT / "silence"
    silence_dir.mkdir(exist_ok=True)
    SAMPLES = int(SR * 3.0)

    current_files = list(silence_dir.glob("*.wav"))
    n_current = len(current_files)
    print(f"  現在の silence: {n_current} 枚")

    if n_current > target:
        print(f"  {n_current} 枚 > {target} 枚なのでダウンサンプリングします")
        downsample_to(current_files, target, OUT / "silence_unused")
        return

    if n_current == target:
        print("  すでに target 枚なので増やしません")
        return

    remaining = target - n_current
    print(f"  silence を {n_current} → {target} 枚に増やします（追加 {remaining} 枚）")

    # 割合: Pink 30%, White 30%, Wind 20%, Pure 20%
    n_pink = int(remaining * 0.3)
    n_white = int(remaining * 0.3)
    n_wind = int(remaining * 0.2)
    n_pure = remaining - n_pink - n_white - n_wind

    print(f"    追加内訳: Pink {n_pink}, White {n_white}, Wind {n_wind}, Pure {n_pure}")

    # ESC-50 wind ファイル一覧
    wind_files = load_wind_file_list()
    if not wind_files:
        # wind が使えない場合は pure に回す
        n_pure += n_wind
        n_wind = 0
        print(f"    wind が見つからないため pure を {n_pure} 枚に変更します")

    # 追加生成
    for i in range(n_pink):
        x = generate_pink_noise(SAMPLES)
        x = scale_to_db(x, (-30, -25))
        fname = silence_dir / f"silence_pink_{uuid.uuid4().hex}.wav"
        sf.write(fname, x, SR)

    for i in range(n_white):
        x = generate_white_noise(SAMPLES)
        x = scale_to_db(x, (-30, -25))
        fname = silence_dir / f"silence_white_{uuid.uuid4().hex}.wav"
        sf.write(fname, x, SR)

    for i in range(n_wind):
        x = random_wind_clip(wind_files, length=SAMPLES)
        x = scale_to_db(x, (-30, -25))  # -30〜-25 dB 相当で弱く
        fname = silence_dir / f"silence_wind_{uuid.uuid4().hex}.wav"
        sf.write(fname, x, SR)

    for i in range(n_pure):
        x = np.zeros(SAMPLES, dtype=np.float32)
        fname = silence_dir / f"silence_pure_{uuid.uuid4().hex}.wav"
        sf.write(fname, x, SR)


def main():
    # 事前カウント
    if not OUT.exists():
        print(f"{OUT} が存在しません。先に抽出スクリプトでデータを作ってください。")
        return

    count_all()

    ans = input(
        f"\n[siren, other, silence] を {TARGET} 枚にそろえるバランス調整を行いますか？ (y/n): "
    ).strip().lower()

    if ans != "y":
        print("バランス調整を行わず終了します。")
        return

    print("\n=== siren のダウンサンプリング ===")
    siren_files = list_wavs("siren")
    downsample_to(siren_files, TARGET, OUT / "siren_unused")

    print("\n=== other のオーバーサンプリング ===")
    other_files = list_wavs("other")
    oversample_to(other_files, TARGET)

    print("\n=== silence の拡張（Pink / White / Wind / Pure） ===")
    augment_silence_to_target(TARGET)

    print("\n=== 調整後の枚数 ===")
    count_all()
    print("完了しました。")


if __name__ == "__main__":
    main()
