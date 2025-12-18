# -*- coding: utf-8 -*-
# save as: train_lstm_mfcc_from_mix.py
#
# UrbanSound8k_ESC50_mix（クラスごとに集約済み）を使用し
# MFCC + RNN（Bi-LSTM）で学習するコード
#
# TrainCRNN_Mix.py の構造に完全準拠（MFCC版）

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau# type: ignore

import seaborn as sns
import random

# ============================================================
# 1) 設定
# ============================================================
DATA_DIR = Path("../../../data/UrbanSound8k_ESC50_mix")
OUT_DIR = Path("../../../Result/runs_lstm_mfcc_mix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 1.5
WIN_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 20              # TrainLSTM.py の 13 → 精度改善のため 40 に拡張
HOP_LENGTH = 512

TEST_SIZE = 0.2
VAL_SIZE = 0.1
EPOCHS = 80
BATCH_SIZE = 32
PATIENCE = 15
SEED = 42


# ============================================================
# 2) 下準備
# ============================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)


def list_all_data(data_dir):
    wavs, labels = [], []
    for cls in sorted(data_dir.iterdir()):
        if cls.is_dir():
            for wav in sorted(cls.glob("*.wav")):
                wavs.append(wav)
                labels.append(cls.name)
    return np.array(wavs), np.array(labels)


def build_label_map(label_txt):
    classes = sorted(list(set(label_txt)))
    return {c: i for i, c in enumerate(classes)}


# ============================================================
# 3) MFCC 生成 & Augmentation
# ============================================================
def wav_to_mfcc(path: Path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) < WIN_SAMPLES:
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))
    else:
        y = y[:WIN_SAMPLES]

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH
    )

    mfcc = mfcc.T  # (time, features)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.astype(np.float32)


def spec_augment_mfcc(mfcc, freq_mask_param=6, time_mask_param=12, p=0.5):
    if np.random.rand() > p:
        return mfcc
    x = mfcc.copy()

    # 周波数方向マスク
    f = np.random.randint(0, freq_mask_param)
    f0 = np.random.randint(0, max(1, N_MFCC - f))
    x[:, f0:f0 + f] = 0

    # 時間方向マスク
    t = np.random.randint(0, time_mask_param)
    t0 = np.random.randint(0, max(1, x.shape[0] - t))
    x[t0:t0 + t, :] = 0

    return x


# ============================================================
# 4) Sequence クラス
# ============================================================
class MFCCSequence(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, n_classes, training=True):
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.training = training
        self.n_classes = n_classes
        self.indexes = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def on_epoch_end(self):
        if self.training:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for i in idxs:
            mfcc = wav_to_mfcc(self.paths[i])
            if self.training:
                mfcc = spec_augment_mfcc(mfcc)
            X.append(mfcc)
            y.append(self.labels[i])

        X = np.stack(X)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


# ============================================================
# 5) LSTM モデル
# ============================================================
def build_lstm_model(time_dim, n_classes):
    inp = layers.Input(shape=(time_dim, N_MFCC))

    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# 6) 学習開始
# ============================================================
def main():
    print("[INFO] データ読込中...")

    paths, labels_txt = list_all_data(DATA_DIR)
    label_map = build_label_map(labels_txt)
    id_to_label = {v: k for k, v in label_map.items()}

    y_all = np.array([label_map[t] for t in labels_txt])

    # 1サンプルで time_dim を測る
    time_dim = wav_to_mfcc(paths[0]).shape[0]
    n_classes = len(label_map)

    # === train / test ===
    train_paths, test_paths, train_y, test_y = train_test_split(
        paths, y_all, test_size=TEST_SIZE,
        stratify=y_all, random_state=SEED
    )

    # === train / val ===
    tr_paths, val_paths, tr_y, val_y = train_test_split(
        train_paths, train_y, test_size=VAL_SIZE,
        stratify=train_y, random_state=SEED
    )

    # class weight
    base_w = compute_class_weight("balanced", classes=np.arange(n_classes), y=tr_y)
    class_weights = {i: w for i, w in enumerate(base_w)}

    # siren 補正（TrainCRNN_Mix.py と同じ）
    for lbl, idx in label_map.items():
        if lbl.lower() == "siren":
            class_weights[idx] = class_weights[idx] * 0.3

    # Generators
    tr_gen = MFCCSequence(tr_paths, tr_y, BATCH_SIZE, n_classes, training=True)
    val_gen = MFCCSequence(val_paths, val_y, BATCH_SIZE, n_classes, training=False)
    test_gen = MFCCSequence(test_paths, test_y, BATCH_SIZE, n_classes, training=False)

    # モデル構築
    model = build_lstm_model(time_dim, n_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(OUT_DIR / "best.keras", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8)
    ]

    history = model.fit(
        tr_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 学習曲線
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.grid(True); plt.title("Loss"); plt.tight_layout()
    plt.savefig(OUT_DIR / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.grid(True); plt.title("Accuracy"); plt.tight_layout()
    plt.savefig(OUT_DIR / "acc_curve.png")
    plt.close()

    # テスト評価
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n=== Test Classification Report ===")
    print(classification_report(test_y, y_pred, target_names=[id_to_label[i] for i in range(n_classes)], digits=4))

    cm = confusion_matrix(test_y, y_pred)
    np.save(OUT_DIR / "cm.npy", cm)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=list(id_to_label.values()),
                yticklabels=list(id_to_label.values()))
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cm.png")
    plt.close()

    model.save(OUT_DIR / "final.keras")
    print(f"[DONE] Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
