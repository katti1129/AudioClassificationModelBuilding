import numpy as np
import librosa
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical# type: ignore
from tensorflow.keras.models import Sequential, load_model# type: ignore
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import japanize_matplotlib

# --- 1. 設定・パラメータ ---
# ◆◆◆ 変更点: データセットのルートディレクトリを指定 ◆◆◆
DATA_DIR = Path('../data/segmented_data_split')
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'validation'
TEST_DIR = DATA_DIR / 'test'

CLASS_NAMES = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
if not CLASS_NAMES:
    raise ValueError(f"訓練データディレクトリ '{TRAIN_DIR}' の中にクラスごとのフォルダが見つかりません．")
NUM_CLASSES = len(CLASS_NAMES)
print(f"クラスを検出: {CLASS_NAMES}")

# パラメータは変更なし
SAMPLE_RATE = 16000
DURATION = 1.5
MAX_LEN_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MFCC = 13
HOP_LENGTH = 512
MAX_LEN_FRAMES = int(np.ceil(MAX_LEN_SAMPLES / HOP_LENGTH))
BATCH_SIZE = 16
EPOCHS = 100

# --- 2. データ拡張パイプラインの定義 (変更なし) ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# --- 3. データジェネレータの作成 (変更なし) ---
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
        return int(np.floor(len(self.file_paths) / self.batch_size))

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
                if self.is_training:
                    wav = augment(samples=wav, sample_rate=SAMPLE_RATE)
                mfccs = librosa.feature.mfcc(y=wav, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
                mfccs = mfccs.T
                if mfccs.shape[0] < MAX_LEN_FRAMES:
                    pad_width = MAX_LEN_FRAMES - mfccs.shape[0]
                    mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
                else:
                    mfccs = mfccs[:MAX_LEN_FRAMES, :]
                X.append(mfccs)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return np.array(X), to_categorical(batch_labels, num_classes=self.num_classes)


# --- 4. データ準備 (よりシンプルに) ---
print("ジェネレータを準備しています．．．")
train_generator = AudioDataGenerator(TRAIN_DIR, BATCH_SIZE, is_training=True)
val_generator = AudioDataGenerator(VAL_DIR, BATCH_SIZE, is_training=False)
test_generator = AudioDataGenerator(TEST_DIR, BATCH_SIZE, is_training=False)

print(f"データ数 -> 訓練: {len(train_generator.file_paths)}件, 検証: {len(val_generator.file_paths)}件, テスト: {len(test_generator.file_paths)}件")


# --- 5. RNN (LSTM) モデルの構築 (変更なし) ---
print("RNN(LSTM)モデルを構築しています．．．")
input_shape = (MAX_LEN_FRAMES, N_MFCC)
model = Sequential([
    Input(shape=input_shape),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.summary()
# --- 6. モデルのコンパイルと学習 ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
             ModelCheckpoint('best_rnn_model.keras', monitor='val_loss', save_best_only=True)]

print("モデルの学習を開始します...")
history = model.fit(train_generator, validation_data=val_generator,
                    epochs=EPOCHS, callbacks=callbacks)

# --- historyオブジェクトから各指標を取得 ---
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# エポック数を取得
epochs = range(1, len(train_acc) + 1)

# --- 1. 正解率(Accuracy)のプロット ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) # 1行2列のグラフの1番目を描画
plt.plot(epochs, train_acc, 'bo-', label='訓練データの正解率 (Training Acc)')
plt.plot(epochs, val_acc, 'ro-', label='検証データの正解率 (Validation Acc)')
plt.title('訓練データと検証データの正解率')
plt.xlabel('エポック数 (Epochs)')
plt.ylabel('正解率 (Accuracy)')
plt.legend()
plt.grid(True)

# --- 2. 損失(Loss)のプロット ---
plt.subplot(1, 2, 2) # 1行2列のグラフの2番目を描画
plt.plot(epochs, train_loss, 'bo-', label='訓練データの損失 (Training Loss)')
plt.plot(epochs, val_loss, 'ro-', label='検証データの損失 (Validation Loss)')
plt.title('訓練データと検証データの損失')
plt.xlabel('エポック数 (Epochs)')
plt.ylabel('損失 (Loss)')
plt.legend()
plt.grid(True)

# グラフを表示
plt.tight_layout() # グラフのレイアウトを調整
plt.show()


# --- 7. モデルの評価 ---
print("\n最適なモデルで検証データに対する評価を行います...")
best_model = load_model('best_rnn_model.keras')
# 検証データではなく、テストデータで評価
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 混同行列と分類レポートもテストデータで作成
print("\n混同行列と分類レポートを作成します．．．")
y_pred = best_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# テストデータ用のジェネレータから真のラベルを取得
y_true = []
for i in range(len(test_generator)):
    _, labels_batch = test_generator[i]
    y_true.extend(np.argmax(labels_batch, axis=1))

cm = confusion_matrix(y_true, y_pred_classes[:len(y_true)])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix on Test Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\n分類レポート (Test Data):")
print(classification_report(y_true, y_pred_classes[:len(y_true)], target_names=CLASS_NAMES))
