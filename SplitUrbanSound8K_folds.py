# save as: split_UrbanSound8K_into_folds.py
#UrbanSound8K_split_safe/
#├─ fold1/ ├─car_horn|engine_idling|siren
#├─ fold2/ ...
#└─ fold10/ ...
#の階層に分ける

import pandas as pd
from pathlib import Path
import shutil

# === パス設定 ===
base_dir = Path("../data/UrbanSound8K")
audio_dir = base_dir / "audio"
metadata_path = base_dir / "metadata" / "UrbanSound8K.csv"

# 出力（foldごと）
out_root = Path("../data/UrbanSound8K_split_safe")
out_root.mkdir(parents=True, exist_ok=True)

# 対象クラス（好きに変更OK）
target_classes = ["car_horn", "engine_idling", "siren"]

# === メタ読み込み & 抽出 ===
meta = pd.read_csv(metadata_path)
meta = meta[meta["class"].isin(target_classes)]

# === foldごとにコピー ===
missing = 0
for _, r in meta.iterrows():
    fold_name = f"fold{int(r['fold'])}"
    src = audio_dir / fold_name / r["slice_file_name"]
    if not src.exists():
        print(f"⚠ Missing: {src}")
        missing += 1
        continue
    dst_dir = out_root / fold_name / r["class"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst_dir / r["slice_file_name"])

print("\n✅ 完了")
print(f"出力先: {out_root.resolve()}")
if missing:
    print(f"⚠ 見つからなかったファイル: {missing} 件")
