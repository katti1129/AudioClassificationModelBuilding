# -*- coding: utf-8 -*-
# save as: train_crnn_sequential_windowed_fold1to9.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau# type: ignore
import random

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import seaborn as sns

# =========================================================
# 1) 設定
# =========================================================
DATA_DIR = Path("../../data/UrbanSound8K_split_4sec")
SAMPLE_RATE = 16000
DURATION = 4.0
WIN_SAMPLES = int(SAMPLE_RATE * DURATION)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
FMIN, FMAX = 20, 8000

EPOCHS = 100
BATCH_SIZE = 32
VAL_SIZE = 0.1
PATIENCE = 20
SEED = 42

# =========================================================
# 2) ユーティリティ
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
set_seed(SEED)

def list_folds(data_dir: Path):
    """fold1〜fold9のみを返す"""
    return [p for p in sorted(data_dir.iterdir())
            if p.is_dir() and p.name.startswith("fold")
            and not p.name.endswith("10")]

def list_wavs_and_labels(fold_dir: Path):
    wavs, labels = [], []
    for cls_dir in sorted(fold_dir.iterdir()):
        if cls_dir.is_dir():
            for wav in cls_dir.glob("*.wav"):
                wavs.append(wav)
                labels.append(cls_dir.name)
    return wavs, labels

def build_label_map(data_dir: Path):
    classes = sorted({cls.name for f in data_dir.iterdir()
                      if f.is_dir() and not f.name.endswith("10")
                      for cls in f.iterdir() if cls.is_dir()})
    return {c: i for i, c in enumerate(classes)}

def wav_to_logmelspec(path: Path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < WIN_SAMPLES:
        y = np.pad(y, (0, WIN_SAMPLES - len(y)))
    else:
        y = y[:WIN_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)

def spec_augment(logmel, freq_mask_param=12, time_mask_param=8, p=0.5):
    if np.random.rand() > p:
        return logmel
    x = logmel.copy()
    f = np.random.randint(0, freq_mask_param + 1)
    f0 = np.random.randint(0, max(1, N_MELS - f))
    x[f0:f0 + f, :] = 0
    t = np.random.randint(0, time_mask_param + 1)
    t0 = np.random.randint(0, max(1, x.shape[1] - t))
    x[:, t0:t0 + t] = 0
    return x

class MelSequence(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, n_classes, training=True):
        self.paths = paths
        self.labels = labels
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

# =========================================================
# 3) モデル構築
# =========================================================
def build_crnn(n_classes: int, time_dim: int):
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
    x = layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# =========================================================
# 4) 学習ループ（fold1〜9のみ）
# =========================================================
def get_time_dim_example():
    folds = [f for f in sorted(DATA_DIR.iterdir())
             if f.is_dir() and f.name.startswith("fold") and not f.name.endswith("10")]
    for f in folds:
        wavs, _ = list_wavs_and_labels(f)
        if wavs:
            return wav_to_logmelspec(wavs[0]).shape[1]
    return 47

def run_fold_training():
    label_map = build_label_map(DATA_DIR)
    n_classes = len(label_map)
    id_to_label = {v: k for k, v in label_map.items()}
    time_dim = get_time_dim_example()

    folds = list_folds(DATA_DIR)  # fold1〜fold9 を取得

    # --- 評価スコア格納 ---
    all_accuracies = []
    all_macro_f1 = []
    all_weighted_f1 = []

    print(f"[Info] Using folds 1–9 for cross-validation (fold10 excluded)")
    print(f"[Info] Input shape = ({N_MELS}, {time_dim}, 1)")

    for i, test_fold in enumerate(folds):
        # --- val fold を test fold の次にローテーションで設定 ---
        val_fold = folds[(i + 1) % len(folds)]

        print("\n" + "="*70)
        print(f"[Fold {i+1}]  Test: {test_fold.name} | Val: {val_fold.name}")
        print("="*70)

        # ====== Test ======
        test_paths, test_labels_txt = list_wavs_and_labels(test_fold)
        y_test = np.array([label_map[t] for t in test_labels_txt])

        # ====== Val ======
        val_paths, val_labels_txt = list_wavs_and_labels(val_fold)
        y_val = np.array([label_map[t] for t in val_labels_txt])

        # ====== Train ======
        train_folds = [f for f in folds if f not in (test_fold, val_fold)]
        train_paths, train_labels_txt = [], []
        for f in train_folds:
            p, l = list_wavs_and_labels(f)
            train_paths.extend(p)
            train_labels_txt.extend(l)
        y_train = np.array([label_map[t] for t in train_labels_txt])

        # ---- class_weight は y_train のみから計算（リーク対策） ----
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=y_train
        )
        class_weights = {i: w for i, w in enumerate(class_weights)}

        # ---- Siren 重みを抑制してリアルタイム過検出を防止 ----
        # siren クラスの index を特定
        for label, idx in label_map.items():
            if label.lower() == "siren":
                siren_idx = idx
                break

        # siren の重みを 0.3倍（＝約70% 削減）
        # ※ 最初は 0.5 でも OK ですが 0.3 が実運用に効果大
        class_weights[siren_idx] = class_weights[siren_idx] * 0.3

        print("[Info] class_weights after siren suppression:", class_weights)

        # ---- Generators ----
        tr_gen = MelSequence(train_paths, y_train.tolist(), BATCH_SIZE, n_classes, True)
        va_gen = MelSequence(val_paths,   y_val.tolist(),   BATCH_SIZE, n_classes, False)
        te_gen = MelSequence(test_paths,  y_test.tolist(),  BATCH_SIZE, n_classes, False)

        # ---- Build model ----
        model = build_crnn(n_classes, time_dim)

        out_dir = Path(f"../../Result/runs_crnn_sequential_fold1to9/{test_fold.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Callbacks (val_lossで監視) ----
        cbs = [
            EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(out_dir / "best.keras", monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=8, min_lr=1e-6, verbose=1)
        ]

        # ---- Train ----
        history = model.fit(
            tr_gen, epochs=EPOCHS, validation_data=va_gen,
            class_weight=class_weights, callbacks=cbs,
            verbose=1
        )

        # ---- Loss Curve ----
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch");
        plt.ylabel("Loss")
        plt.title(f"Loss Curve ({test_fold.name})")
        plt.grid(True);
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()

        # ---- Accuracy Curve ----
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch");
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curve ({test_fold.name})")
        plt.grid(True);
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "acc_curve.png")
        plt.close()

        # ---- Evaluate ----
        print(f"\n[Eval] Testing on {test_fold.name}")
        y_pred_prob = model.predict(te_gen, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # ---- classification report ----
        print(classification_report(
            y_test, y_pred,
            target_names=[id_to_label[i] for i in range(n_classes)],
            digits=4))

        # ---- calc scores ----
        acc = accuracy_score(y_test, y_pred)
        p,r,f1,_ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        _,_,f1w,_ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        all_accuracies.append(acc)
        all_macro_f1.append(f1)
        all_weighted_f1.append(f1w)

        print(f"[Fold {i+1}] accuracy={acc:.4f}, macro-F1={f1:.4f}, weighted-F1={f1w:.4f}")

        # ---- save confusion matrices ----
        cm = confusion_matrix(y_test, y_pred)
        np.save(out_dir / "cm.npy", cm)

        # --- normalized confusion matrix ---
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
        np.save(out_dir / "cm_norm.npy", cm_norm)

        # ---- draw & save normalization heatmap ----
        import seaborn as sns
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[id_to_label[i] for i in range(n_classes)],
                    yticklabels=[id_to_label[i] for i in range(n_classes)])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Norm) - {test_fold.name}")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix_normalized.png")
        plt.close()

        model.save(out_dir / "final.keras")
        print(f"[Saved results to] {out_dir.resolve()}")

    # ===== CV summary =====
    print("\n" + "="*70)
    print("[Cross Validation Summary]")
    print("="*70)
    print(f"Mean Accuracy    : {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Mean Macro-F1    : {np.mean(all_macro_f1):.4f} ± {np.std(all_macro_f1):.4f}")
    print(f"Mean Weighted-F1 : {np.mean(all_weighted_f1):.4f} ± {np.std(all_weighted_f1):.4f}")
    print("="*70)


if __name__ == "__main__":
    run_fold_training()
