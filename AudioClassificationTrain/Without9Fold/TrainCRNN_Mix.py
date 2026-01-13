# -*- coding: utf-8 -*-
# save as: train_crnn_holdout_from_mix.py
#
# UrbanSound8k_ESC50_mix（クラスごとに集約済み）から
# train / val / test を stratified split
# → CV と同じ CRNN + Attention 条件で学習

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau# type: ignore

import seaborn as sns
import random

# ============================================================
# 1) 設定
# ============================================================
DATA_DIR = Path("../../../data/UrbanSound8K_ESC50_1p")
OUT_DIR = Path("../../../Result/runs_crnn_slidwindow_1p_holdout_mix")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = 1.0
WIN_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000

TEST_SIZE = 0.2
VAL_SIZE = 0.1
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5
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
    """クラスフォルダから wav と labels を収集"""
    wavs, labels = [], []
    for cls_dir in sorted(data_dir.iterdir()):
        if cls_dir.is_dir():
            cls_name = cls_dir.name
            for wav in sorted(cls_dir.glob("*.wav")):
                wavs.append(wav)
                labels.append(cls_name)
    return np.array(wavs), np.array(labels)


def build_label_map(label_txt):
    classes = sorted(list(set(label_txt.tolist())))
    return {c: i for i, c in enumerate(classes)}


def wav_to_logmelspec(path: Path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) < WIN_SAMPLES:
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))
    else:
        y = y[:WIN_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


def spec_augment(logmel, freq_mask_param=12, time_mask_param=8, p=0.5):
    if np.random.rand() > p:
        return logmel

    x = logmel.copy()

    # freq mask
    f = np.random.randint(0, freq_mask_param + 1)
    f0 = np.random.randint(0, max(1, N_MELS - f))
    x[f0:f0 + f, :] = 0

    # time mask
    t = np.random.randint(0, time_mask_param + 1)
    t0 = np.random.randint(0, max(1, x.shape[1] - t))
    x[:, t0:t0 + t] = 0

    return x


# === Attention Layer ===
@tf.keras.utils.register_keras_serializable()
class ReduceSumLayer(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def attention_pooling(inputs):
    score = layers.Dense(1, activation="tanh")(inputs)
    weights = layers.Softmax(axis=1)(score)
    weighted = layers.Multiply()([inputs, weights])
    output = ReduceSumLayer(axis=1)(weighted)
    return output


class MelSequence(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, n_classes, training=True):
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.training = training
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
            logmel = wav_to_logmelspec(self.paths[i])
            if self.training:
                logmel = spec_augment(logmel)
            X.append(np.expand_dims(logmel, -1))
            y.append(self.labels[i])
        X = np.stack(X)
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


def build_crnn(n_classes, time_dim):
    inp = layers.Input(shape=(N_MELS, time_dim, 1))

    x = layers.Conv2D(32, (3,3), padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D((2,1))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Permute((2,1,3))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = attention_pooling(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# 3) 学習開始
# ============================================================
def main():
    print("[INFO] データ読込中...")

    paths, labels_txt = list_all_data(DATA_DIR)
    label_map = build_label_map(labels_txt)
    id_to_label = {v: k for k, v in label_map.items()}

    y_all = np.array([label_map[t] for t in labels_txt])

    # time_dim 確認
    time_dim = wav_to_logmelspec(paths[0]).shape[1]

    # === train / test split ===
    train_paths, test_paths, train_y, test_y = train_test_split(
        paths, y_all,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all
    )

    # === train / val split ===
    tr_paths, val_paths, tr_y, val_y = train_test_split(
        train_paths, train_y,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=train_y
    )

    n_classes = len(label_map)

    # === class_weight ===
    base_w = compute_class_weight("balanced", classes=np.arange(n_classes), y=tr_y)
    class_weights = {i: w for i, w in enumerate(base_w)}

    # siren だけ補正：CVと全く同じ条件
    for lbl, idx in label_map.items():
        if lbl.lower() == "siren":
            class_weights[idx] = class_weights[idx] * 0.3

    # === Generators ===
    tr_gen = MelSequence(tr_paths, tr_y, BATCH_SIZE, n_classes, training=True)
    val_gen = MelSequence(val_paths, val_y, BATCH_SIZE, n_classes, training=False)
    test_gen = MelSequence(test_paths, test_y, BATCH_SIZE, n_classes, training=False)

    # === モデル構築 ===
    model = build_crnn(n_classes, time_dim)
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

    # === 学習曲線 ===
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

    # === テスト評価 ===
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

    # === 最終保存 ===
    model.save(OUT_DIR / "final.keras")
    print(f"[DONE] Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
