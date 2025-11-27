import random
from pathlib import Path
import shutil

N_FOLDS = 5
SRC_ROOT = Path("../../data/combined_wav_16k_mono")  # 元データ
OUT_ROOT = Path("../../data/combined_wav_16k_mono_folds_k5")  # 元のままfold分割
CLASSES = ["siren", "other", "silence"]

random.seed(42)

def main():
    for fold in range(1, N_FOLDS + 1):
        for cls in CLASSES:
            (OUT_ROOT / f"fold{fold}" / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        src_dir = SRC_ROOT / cls
        files = sorted(src_dir.glob("*.wav"))

        random.shuffle(files)
        print(f"{cls}: {len(files)} 枚を {N_FOLDS}-fold に分割中")

        for idx, wav in enumerate(files):
            fold_id = (idx % N_FOLDS) + 1
            dst = OUT_ROOT / f"fold{fold_id}" / cls / wav.name
            shutil.copyfile(wav, dst)

    print("完了：元 WAV の 5-fold 分割終了")

if __name__ == "__main__":
    main()
