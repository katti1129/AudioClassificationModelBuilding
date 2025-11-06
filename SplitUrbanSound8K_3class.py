import pandas as pd
import shutil
from pathlib import Path

# --- 1. パス設定 ---
base_dir = Path("../data/UrbanSound8K")                # UrbanSound8Kのルート
audio_dir = base_dir / "audio"                 # 各foldフォルダが入っている場所
metadata_path = base_dir / "metadata" / "UrbanSound8K.csv"
output_root = Path("../data/UrbanSound8K_split")       # 出力ルートフォルダ
train_dir = output_root / "train_data"         # fold1～9を格納
real_time_dir = output_root / "real_time_eval" # fold10を格納

# --- 2. 使うクラスを指定 ---
target_classes = ["car_horn", "engine_idling", "siren"]

# --- 3. メタデータ読み込み ---
meta = pd.read_csv(metadata_path)

# --- 4. 対象クラスのみ抽出 ---
meta = meta[meta["class"].isin(target_classes)]

# --- 5. foldで分割 ---
train_data = meta[meta["fold"] < 10]   # fold1〜9
realtime_data = meta[meta["fold"] == 10]  # fold10

print(f"fold1〜9 (学習・検証用): {len(train_data)} ファイル")
print(f"fold10 (リアルタイム検証用): {len(realtime_data)} ファイル")

# --- 6. コピー関数 ---
def copy_files(df, target_root):
    for _, row in df.iterrows():
        fold = f"fold{row['fold']}"
        src = audio_dir / fold / row["slice_file_name"]
        dst_class_dir = target_root / row["class"]
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst_class_dir / row["slice_file_name"])

# --- 7. データを分類してコピー ---
print("\n📦 データをコピー中...")
copy_files(train_data, train_dir)
copy_files(realtime_data, real_time_dir)

print("\n✅ 分類完了！")
print(f"学習・検証データ: {train_dir.resolve()}")
print(f"リアルタイム検証用: {real_time_dir.resolve()}")
