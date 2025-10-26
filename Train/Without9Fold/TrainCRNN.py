# 02_train_crnn_with_evaluation.py
# 役割: 訓練・検証データで学習し，学習後にグラフ表示とテストデータでの最終評価を行う

import numpy as np
import librosa
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical# type: ignore
from tensorflow.keras.models import Sequential, load_model# type: ignore
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Activation, Reshape, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import japanize_matplotlib # 日本語表示のため

# --- 1. 設定・パラメータ ---
# ◆ スライディングウィンドウで生成したデータが入ったフォルダを指定
DATA_DIR = Path('../data/segmented_data_split')
MODEL_SAVE_PATH = 'best_crnn_model.keras'

# 各セットのディレクトリパス
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'validation'
TEST_DIR = DATA_DIR / 'test'

CLASS_NAMES = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
print(f"クラスを検出: {CLASS_NAMES}")

SAMPLE_RATE = 16000
DURATION = 1.5
MAX_LEN_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 100

# --- 2. データ拡張パイプライン ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# --- 3. データジェネレータ ---
class AudioDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size, is_training=True):
        self.class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        self.class_names = [p.name for p in self.class_dirs]
        self.file_paths, self.labels = self._load_filepaths_and_labels()
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_classes = len(self.class_names)
        self.on_epoch_end()

    def _load_filepaths_and_labels(self):
        files, labels = [], []
        for i, class_dir in enumerate(self.class_dirs):
            for file_path in class_dir.glob('*.wav'):
                files.append(file_path)
                labels.append(i)
        return np.array(files), np.array(labels)

    def __len__(self):
        # 最後の半端なバッチも処理するようにceil（切り上げ）を使う
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self._generate_data(batch_files, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.is_training:
            indices = np.arange(len(self.file_paths))
            np.random.shuffle(indices)
            self.file_paths = self.file_paths[indices]
            self.labels = self.labels[indices]

    def _generate_data(self, batch_files, batch_labels):
        X = []
        for file_path in batch_files:
            try:
                wav, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                if len(wav) > MAX_LEN_SAMPLES: wav = wav[:MAX_LEN_SAMPLES]
                else: wav = np.pad(wav, (0, MAX_LEN_SAMPLES - len(wav)), mode='constant')
                if self.is_training: wav = augment(samples=wav, sample_rate=SAMPLE_RATE)
                melspec = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                log_melspec = librosa.power_to_db(melspec, ref=np.max)
                log_melspec = np.expand_dims(log_melspec, axis=-1)
                X.append(log_melspec)
            except Exception as e: print(f"Error processing {file_path}: {e}")
        return np.array(X), to_categorical(batch_labels, num_classes=self.num_classes)

# --- 4. データ準備 ---
print("ジェネレータを準備しています．．．")
train_generator = AudioDataGenerator(TRAIN_DIR, BATCH_SIZE, is_training=True)
val_generator = AudioDataGenerator(VAL_DIR, BATCH_SIZE, is_training=False)
test_generator = AudioDataGenerator(TEST_DIR, BATCH_SIZE, is_training=False)
print(f"訓練: {len(train_generator.file_paths)}件, 検証: {len(val_generator.file_paths)}件, テスト: {len(test_generator.file_paths)}件")

# --- 5. CRNNモデルの構築 ---
print("CRNNモデルを構築しています．．．")
sample_batch, _ = train_generator[0]
input_shape = sample_batch.shape[1:]
model = Sequential([
    Input(shape=input_shape),
    Conv2D(32, (3, 3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding='same'), BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),
    Reshape((-1, 128 * (N_MELS // 8))),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'), Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.summary()

# --- 6. モデルのコンパイルと学習 ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
             ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)]
print("モデルの学習を開始します．．．")
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=callbacks)

# ◆◆◆ここからが追加・変更点◆◆◆

# --- 7. 学習曲線のプロット ---
print("\n学習曲線をプロットします．．．")
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='訓練データの正解率 (Train)')
plt.plot(epochs, val_acc, 'ro-', label='検証データの正解率 (Val)')
plt.title('訓練・検証データの正解率')
plt.xlabel('エポック数')
plt.ylabel('正解率')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='訓練データの損失 (Train)')
plt.plot(epochs, val_loss, 'ro-', label='検証データの損失 (Val)')
plt.title('訓練・検証データの損失')
plt.xlabel('エポック数')
plt.ylabel('損失')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 8. テストデータでの最終評価 ---
print(f"\n最適なモデル({MODEL_SAVE_PATH})でテストデータに対する最終評価を行います．．．")
best_model = load_model(MODEL_SAVE_PATH)

test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 混同行列と分類レポートの表示
print("\n混同行列と分類レポートを作成します．．．")
y_pred = best_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.labels[:len(y_pred_classes)]

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('テストデータに対する混同行列 (Confusion Matrix on Test Data)')
plt.xlabel('予測ラベル (Predicted Label)')
plt.ylabel('正解ラベル (True Label)')
plt.show()

print("\n分類レポート (Test Data):")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

# --- 9. エラー分析：誤分類したファイルを抽出 ---
print("\n誤って分類されたファイルを抽出します．．．")
test_files = test_generator.file_paths
misclassified_files = []
for i in range(len(y_true)):
    if y_true[i] != y_pred_classes[i]:
        misclassified_info = {
            "file_path": test_files[i],
            "true_label": CLASS_NAMES[y_true[i]],
            "predicted_label": CLASS_NAMES[y_pred_classes[i]]
        }
        misclassified_files.append(misclassified_info)

if not misclassified_files:
    print("誤って分類されたファイルはありませんでした．")
else:
    print(f"合計 {len(misclassified_files)} 個のファイルが誤って分類されました．")
    for info in misclassified_files:
        print(f"  - ファイル: {info['file_path']}")
        print(f"    - 正解: {info['true_label']}")
        print(f"    - 予測: {info['predicted_label']}\n")