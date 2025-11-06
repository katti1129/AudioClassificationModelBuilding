from pathlib import Path
import os

DATA_DIR = Path("../data/UrbanSound8K_split_4sec")  #  <- あなたの環境に合わせて

def count_files_per_fold(data_dir: Path):
    folds = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    for fold in folds:
        print(f"\n[{fold.name}]")
        classes = sorted([c for c in fold.iterdir() if c.is_dir()])
        for cls in classes:
            wav_count = len(list(cls.glob("*.wav")))
            print(f"  {cls.name}: {wav_count}")

if __name__ == "__main__":
    count_files_per_fold(DATA_DIR)
