# 02_extract_features_split.py

import numpy as np
import librosa
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# --- 設定 ---
SEGMENT_DATA_DIR = Path('../data/segmented_data_split')  # ◆ 変更点: 新しいフォルダを指定
OUTPUT_DIR = Path('../features')

SAMPLE_RATE = 16000

# --- YAMNetモデルのロード ---
print("YAMNetモデルをロードしています．．．")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# --- メイン処理 ---
OUTPUT_DIR.mkdir(exist_ok=True)

# ◆ 変更点: train, validation, test の各セットをループで処理する
for dataset_type in ['train', 'validation', 'test']:
    print(f"\nデータセット '{dataset_type}' の特徴量抽出を開始します．．．")

    current_set_dir = SEGMENT_DATA_DIR / dataset_type
    if not current_set_dir.exists():
        print(f"ディレクトリ '{current_set_dir}' が見つかりません．スキップします．")
        continue

    CLASS_NAMES = sorted([p.name for p in current_set_dir.iterdir() if p.is_dir()])
    all_features = []
    all_labels = []

    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = current_set_dir / class_name
        wav_files = list(class_dir.glob('*.wav'))

        print(f"  クラス '{class_name}' ({len(wav_files)}個) を処理中．．．")

        for wav_file in tqdm(wav_files):
            try:
                wav, _ = librosa.load(wav_file, sr=SAMPLE_RATE, mono=True)

                _, embeddings, _ = yamnet_model(wav)

                all_features.append(embeddings.numpy())
                all_labels.append(i)

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

    # ◆ 変更点: セットごとに特徴量ファイルを保存
    output_npz_file = OUTPUT_DIR / f'yamnet_features_{dataset_type}.npz'
    np.savez(
        output_npz_file,
        features=np.array(all_features, dtype=object),
        labels=np.array(all_labels),
        class_names=np.array(CLASS_NAMES)
    )
    print(f"'{dataset_type}' の特徴量を '{output_npz_file}' に保存しました．")

print("\n全ての特徴量抽出が完了しました．")