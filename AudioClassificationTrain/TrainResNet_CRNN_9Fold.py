# ==========================================================
# ResNet + CRNN モデル（特徴抽出＋時系列処理）
# 9分割交差検証による学習・評価コード
# ==========================================================

import numpy as np
import librosa
from pathlib import Path
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from collections import Counter
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

import tensorflow as tf
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.layers import (Input, Reshape, Bidirectional, LSTM,
                                     Dense, Dropout)
from tensorflow.keras.applications import ResNet50# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# --- 1. 設定・パラメータ ---
DATA_DIR = Path('../data/segmented_data_split/train')  # 全データをこの中に統合
SAMPLE_RATE = 16000
DURATION = 1.5
MAX_LEN_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
EPOCHS = 50
BATCH_SIZE = 16
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. クラス検出 ---
CLASS_NAMES = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)
print(f"検出クラス: {CLASS_NAMES}")

# --- 3. データ拡張 ---
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# --- 4. データ読み込み ---
print("音声データを読み込み中...")
X, y = [], []
for i, class_name in enumerate(CLASS_NAMES):
    for wav_path in (DATA_DIR / class_name).glob('*.wav'):
        wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        if len(wav) > MAX_LEN_SAMPLES:
            wav = wav[:MAX_LEN_SAMPLES]
        else:
            wav = np.pad(wav, (0, MAX_LEN_SAMPLES - len(wav)), mode='constant')

        # メルスペクトログラム
        melspec = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # ResNet用に3チャンネルへ拡張
        log_melspec_3ch = np.stack([log_melspec, log_melspec, log_melspec], axis=-1)

        X.append(log_melspec_3ch)
        y.append(i)

X = np.array(X)
y = np.array(y)
print(f"読み込み完了: X={X.shape}, y={y.shape}")

# --- 5. モデル構築関数 ---
def build_resnet_crnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # ResNet50を特徴抽出器として利用（ImageNet重みあり）
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # ResNet出力をRNNに渡すため整形
    x = base_model.output
    reshape_dim = x.shape[1] * x.shape[2]
    feature_dim = x.shape[3]
    x = Reshape((reshape_dim, feature_dim))(x)

    # RNN層
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)

    # 全結合層
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.6)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 6. 9分割交差検証 ---
skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=SEED)
fold_results = []
y_true_all, y_pred_all = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n========== Fold {fold}/9 ==========")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # クラス重み
    cnt = Counter(y_train)
    class_weight = {cls: len(y_train) / (NUM_CLASSES * cnt[cls]) for cls in cnt}

    model = build_resnet_crnn_model(X.shape[1:], NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        ModelCheckpoint(f"resnet_crnn_fold{fold:02d}.keras", monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2
    )

    # foldごとの評価
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_val, y_pred)
    f1m = f1_score(y_val, y_pred, average='macro')
    print(f"[Fold {fold}] Accuracy={acc:.4f}, F1-macro={f1m:.4f}")

    fold_results.append({"fold": fold, "acc": acc, "f1": f1m})
    y_true_all.append(y_val)
    y_pred_all.append(y_pred)

# --- 7. 総合評価 ---
y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

accs = np.array([r["acc"] for r in fold_results])
f1s = np.array([r["f1"] for r in fold_results])

print("\n===== 9-Fold Summary =====")
print(f"Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")
print(f"F1-macro: {f1s.mean():.4f} ± {f1s.std():.4f}")

print("\nClassification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (9-Fold Overall)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
