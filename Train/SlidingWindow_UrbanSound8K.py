# save as: apply_sliding_window_per_fold.py
import librosa, soundfile as sf
from pathlib import Path

# 入力: ①の出力
IN_ROOT = Path("../../data/UrbanSound8K_split_safe")
# 出力: スライド後（fold構造を維持したまま）
OUT_ROOT = Path("../../data/UrbanSound8K_windowed_safe")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
WINDOW = 1.5   # 秒
STEP = 0.75    # 秒
WIN_SAMPLES = int(SAMPLE_RATE * WINDOW)
STEP_SAMPLES = int(SAMPLE_RATE * STEP)

for fold_dir in sorted(IN_ROOT.iterdir()):
    if not fold_dir.is_dir():
        continue
    for class_dir in sorted(fold_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        out_class = OUT_ROOT / fold_dir.name / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        for wav_path in class_dir.glob("*.wav"):
            wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            n = 0
            for start in range(0, len(wav) - WIN_SAMPLES + 1, STEP_SAMPLES):
                seg = wav[start:start+WIN_SAMPLES]
                out_path = out_class / f"{wav_path.stem}_seg{n}.wav"
                sf.write(out_path, seg, SAMPLE_RATE)
                n += 1

print("✅ スライド適用完了")
print(f"出力先: {OUT_ROOT.resolve()}")
