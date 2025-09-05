#transformers、tensorflowとtorchのバージョンが合わない

import tensorflow as tf
from transformers import AutoFeatureExtractor, TFAutoModelForAudioClassification
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.metrics import SparseCategoricalAccuracy  # type: ignore
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 設定 ---
DATA_DIR = Path('../../data/ESC-50-master/organized/segmented_data_split')
MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5

# --- 2. データセットの準備 ---

train_dir = DATA_DIR / 'train'
CLASS_NAMES = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

print(f"{NUM_CLASSES}個のクラスを検出しました: {', '.join(CLASS_NAMES)}")


def load_audio_paths_and_labels(data_dir):
    paths = []
    labels = []
    for class_name, label_id in label2id.items():
        class_dir = Path(data_dir) / class_name
        for file_path in class_dir.glob('*.wav'):
            paths.append(str(file_path))
            labels.append(label_id)
    return paths, labels


train_paths, train_labels = load_audio_paths_and_labels(DATA_DIR / 'train')
val_paths, val_labels = load_audio_paths_and_labels(DATA_DIR / 'validation')
test_paths, test_labels = load_audio_paths_and_labels(DATA_DIR / 'test')

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)


def data_generator(paths, labels, batch_size):
    num_samples = len(paths)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_paths = [paths[j] for j in batch_indices]
            batch_labels = [labels[j] for j in batch_indices]

            raw_audios = [librosa.load(p, sr=feature_extractor.sampling_rate)[0] for p in batch_paths]

            inputs = feature_extractor(raw_audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="tf")

            yield inputs, tf.constant(batch_labels)


train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_paths, train_labels, BATCH_SIZE),
    output_signature=(
        {'input_values': tf.TensorSpec(shape=(None, None), dtype=tf.float32)},
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_paths, val_labels, BATCH_SIZE),
    output_signature=(
        {'input_values': tf.TensorSpec(shape=(None, None), dtype=tf.float32)},
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(test_paths, test_labels, BATCH_SIZE),
    output_signature=(
        {'input_values': tf.TensorSpec(shape=(None, None), dtype=tf.float32)},
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# --- 3. モデルの構築と学習 ---
print("\n事前学習済みASTモデルをロードしています...")
# ★★★ PyTorchの重みをTensorFlowモデルとして読み込むために from_pt=True を追加 ★★★
model = TFAutoModelForAudioClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_CLASSES,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
    from_pt=True,
)

optimizer = Adam(learning_rate=LEARNING_RATE)
loss = SparseCategoricalCrossentropy(from_logits=True)
metric = SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print("\nモデルのファインチューニングを開始します...")
history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=val_dataset,
    validation_steps=len(val_paths) // BATCH_SIZE,
    epochs=NUM_EPOCHS,
)

# --- 4. 評価 ---
print("\nテストデータでモデルを評価します...")
test_loss, test_acc = model.evaluate(test_dataset, steps=len(test_paths) // BATCH_SIZE)
print(f'\nTest accuracy: {test_acc:.4f}')


# --- 5. 学習履歴のグラフ化 ---
def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history.history['loss'])))
    plt.legend()
    plt.grid(True)
    plt.show()


print("\n学習履歴のグラフを表示します...")
plot_history(history)

# --- 6. モデルの保存 ---
model.save_pretrained("./ast_finetuned_model")
feature_extractor.save_pretrained("./ast_finetuned_model")
print("\nファインチューニング済みモデルを ./ast_finetuned_model に保存しました。")