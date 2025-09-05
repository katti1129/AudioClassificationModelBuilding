import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import Sequence, to_categorical# type: ignore
from tensorflow.keras.models import Model, load_model# type: ignore
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout,Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. 設定・パラメータ ---
DATA_DIR = Path('./data')
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
print(f"クラスを検出: {CLASS_NAMES}")

# YAMNetが要求する音声パラメータ
SAMPLE_RATE = 16000
DURATION = 5
MAX_LEN_SAMPLES = SAMPLE_RATE * DURATION

BATCH_SIZE = 16
EPOCHS = 50  # EarlyStoppingを使うので多めに設定

# --- 2. データ拡張パイプラインの定義 ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])


# --- 3. データジェネレータの作成 (YAMNet転移学習用) ---
class TransferLearningDataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, is_training=True):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        # YAMNetは生の波形データを入力とする
        X, y = self._generate_waveforms(batch_files, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.is_training:
            indices = np.arange(len(self.file_paths))
            np.random.shuffle(indices)
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def _generate_waveforms(self, batch_files, batch_labels):
        X = []
        for file_path in batch_files:
            try:
                wav, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

                if len(wav) < MAX_LEN_SAMPLES:
                    wav = np.pad(wav, (0, MAX_LEN_SAMPLES - len(wav)), mode='constant')
                else:
                    wav = wav[:MAX_LEN_SAMPLES]

                if self.is_training:
                    wav = augment(samples=wav, sample_rate=SAMPLE_RATE)

                X.append(wav)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        return np.array(X), to_categorical(batch_labels, num_classes=NUM_CLASSES)


# --- 4. データ準備 ---
print("データセットを読み込んでいます...")
all_files = []
all_labels = []
for i, class_name in enumerate(CLASS_NAMES):
    class_dir = DATA_DIR / class_name
    for file_path in class_dir.glob('*.wav'):
        all_files.append(file_path)
        all_labels.append(i)
print(f"合計 {len(all_files)} 個のファイルが見つかりました。")

train_files, val_files, train_labels, val_labels = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

train_generator = TransferLearningDataGenerator(train_files, train_labels, BATCH_SIZE, is_training=True)
val_generator = TransferLearningDataGenerator(val_files, val_labels, BATCH_SIZE, is_training=False)

# --- 5. 転移学習モデルの構築（最終修正版） ---
print("YAMNetを使った転移学習モデルを構築しています...")

def build_transfer_model(num_classes):
    # YAMNetをKerasのレイヤーとしてロード。これが最も標準的な方法。
    yamnet_layer = hub.KerasLayer(
        "https://tfhub.dev/google/yamnet/1",
        trainable=False,  # YAMNetの重みは学習させない
        name='yamnet'
    )

    # モデルの入力を定義
    input_tensor = Input(shape=(MAX_LEN_SAMPLES,), dtype=tf.float32, name='input_waveform')

    # YAMNetレイヤーに入力を通し、出力を辞書として受け取る
    yamnet_outputs = yamnet_layer(input_tensor)

    # 'embeddings' というキーを指定して、目的の特徴量を取得
    embeddings = yamnet_outputs['embeddings']

    # 後段のRNN部分
    x = Bidirectional(LSTM(64, return_sequences=True))(embeddings)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)

    # 入力と出力を接続してモデルを定義
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

model = build_transfer_model(NUM_CLASSES)
model.summary()

# --- 6. モデルのコンパイルと学習 ---
# (このセクションは変更なし)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
             ModelCheckpoint('best_transfer_model.keras', monitor='val_loss', save_best_only=True)]

print("モデルの学習を開始します...")
# (fitの呼び出しも変更なし)
history = model.fit(train_generator, validation_data=val_generator,
                    epochs=EPOCHS, callbacks=callbacks)

# --- 7. モデルの評価 ---
# (評価部分は元のコードと同じなので省略。ただし、モデル読み込みのファイル名を変更)
# best_model = load_model('best_transfer_model.keras')
# ... (混同行列や分類レポートの表示) ...