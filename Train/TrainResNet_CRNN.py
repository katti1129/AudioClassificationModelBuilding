# 02_train_resnet_crnn.py
# 役割: ResNetを特徴抽出器として使用するCRNNモデルを学習・評価する

import numpy as np
import librosa
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical  # type: ignore
from tensorflow.keras.models import Model, load_model  # type: ignore
from tensorflow.keras.layers import Input, Reshape, Bidirectional, LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import japanize_matplotlib  # 日本語表示のため

# --- 1. 設定・パラメータ ---
DATA_DIR = Path('../../data/ESC-50-master/organized/segmented_data_split')
MODEL_SAVE_PATH = '../model/best_resnet_crnn_model.keras'

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
BATCH_SIZE = 8
EPOCHS = 50

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
                if len(wav) > MAX_LEN_SAMPLES:
                    wav = wav[:MAX_LEN_SAMPLES]
                else:
                    wav = np.pad(wav, (0, MAX_LEN_SAMPLES - len(wav)), mode='constant')
                if self.is_training: wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

                melspec = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
                                                         hop_length=HOP_LENGTH)
                log_melspec = librosa.power_to_db(melspec, ref=np.max)

                # ★★★ ResNet用にスペクトログラムを3チャンネルに変換 ★★★
                log_melspec_3ch = np.stack([log_melspec, log_melspec, log_melspec], axis=-1)

                X.append(log_melspec_3ch)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return np.array(X), to_categorical(batch_labels, num_classes=self.num_classes)


# --- 4. データ準備 ---
print("ジェネレータを準備しています．．．")
train_generator = AudioDataGenerator(TRAIN_DIR, BATCH_SIZE, is_training=True)
val_generator = AudioDataGenerator(VAL_DIR, BATCH_SIZE, is_training=False)
test_generator = AudioDataGenerator(TEST_DIR, BATCH_SIZE, is_training=False)
print(
    f"訓練: {len(train_generator.file_paths)}件, 検証: {len(val_generator.file_paths)}件, テスト: {len(test_generator.file_paths)}件")

# --- 5. ResNet-CRNNモデルの構築 ---
print("ResNet-CRNNモデルを構築しています．．．")
sample_batch, _ = train_generator[0]
input_shape = sample_batch.shape[1:]

# ★★★ ここからがResNetモデルの構築部分 ★★★
# 入力層
inputs = Input(shape=input_shape)

# ResNet50を特徴抽出器として利用 (include_top=Falseで全結合層を除外)
# weights='imagenet'で事前学習済みの重みをロード
resnet_base = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

# ResNetの出力を取得
x = resnet_base.output

# ResNetの出力形状を確認 (例: (None, 4, 1, 2048))
# この形状をRNNが扱えるシーケンス形式 (None, タイムステップ, 特徴数) に変換
# ここでは空間的な次元(高さと幅)を時間ステップとして平坦化
# resnet_base.output.shape[1] = 高さ, [2] = 幅, [3] = チャンネル数
reshape_dim = resnet_base.output.shape[1] * resnet_base.output.shape[2]
features_dim = resnet_base.output.shape[3]
x = Reshape((reshape_dim, features_dim))(x)

# RNN部分 (元のコードと同じ)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Bidirectional(LSTM(64))(x)

# 分類ヘッド部分 (元のコードと同じ)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# モデル全体を定義
model = Model(inputs=inputs, outputs=outputs)
# ★★★ ここまでがResNetモデルの構築部分 ★★★

model.summary()

# --- 6. モデルのコンパイルと学習 ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
             ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)]
print("モデルの学習を開始します．．．")
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=callbacks)

# --- 7. 学習曲線のプロット ---
print("\n学習曲線をプロットします．．．")
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, 'bo-', label='訓練データの正解率 (Train)')
plt.plot(epochs_range, val_acc, 'ro-', label='検証データの正解率 (Val)')
plt.title('訓練・検証データの正解率')
plt.xlabel('エポック数')
plt.ylabel('正解率')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, 'bo-', label='訓練データの損失 (Train)')
plt.plot(epochs_range, val_loss, 'ro-', label='検証データの損失 (Val)')
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
