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
import sys
from datetime import datetime
import gc  # ★ 追加

# =========================================================
# 1) 設定
# =========================================================
DATA_DIR = Path("../../data/clean_folds_5fold")
SAMPLE_RATE = 16000
DURATION = 1.5
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
    """fold1〜fold4 のみ（fold5 は Unity 用として除外）"""
    return [data_dir / f"fold{i}" for i in range(1, 5)]


# ==========================================
# 修正箇所: list_wavs_and_labels
# ==========================================
def list_wavs_and_labels(fold_dir: Path):
    wavs, labels = [], []
    # ディレクトリもソート
    for cls_dir in sorted(fold_dir.iterdir()):
        if cls_dir.is_dir():
            # ★ 修正: ファイル名をソートして取得（再現性確保）
            for wav in sorted(cls_dir.glob("*.wav")):
                wavs.append(wav)
                labels.append(cls_dir.name)
    return wavs, labels

def build_label_map(data_dir: Path):
    """fold1〜fold4 内のクラス名を収集してラベルマップを作成"""
    classes = set()
    for i in range(1, 5):
        fold_dir = data_dir / f"fold{i}"
        if fold_dir.is_dir():
            for cls in fold_dir.iterdir():
                if cls.is_dir():
                    classes.add(cls.name)
    classes = sorted(classes)
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

# === Attention Pooling層 ===
def attention_pooling(inputs):
    # 各時間ステップのスコアを計算
    score = layers.Dense(1, activation="tanh")(inputs)

    # Keras版Softmax層（tf.nn.softmaxではなくlayers.Softmax）
    weights = layers.Softmax(axis=1)(score)

    # 重みをかけて加重平均
    weighted = layers.Multiply()([inputs, weights])

    # Keras Lambda層でreduce_sum
    output = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)

    return output


def run_fold_training():
    # === ログ開始 ===
    log_path = Path("../../Result/runs_crnn_fold1to4_train_eval/log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout = TeeLogger(log_path)

    print("=== Training Start ===")
    print("Timestamp:", datetime.now())



class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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

    # --- 時間軸方向に変換 ---
    x = layers.Permute((2,1,3))(x)
    x = layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)

    # --- RNN部分 ---
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # === Attention Poolingに置き換え ===
    x = attention_pooling(x)
    x = layers.Dropout(0.4)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# =========================================================
# 4) 学習ループ（fold1〜4のみ）
# =========================================================
def get_time_dim_example():
    """fold1〜fold4 の中から1ファイルを読み取り time_dim を取得"""
    for i in range(1, 5):
        fold_dir = DATA_DIR / f"fold{i}"
        if fold_dir.is_dir():
            wavs, _ = list_wavs_and_labels(fold_dir)
            if wavs:
                return wav_to_logmelspec(wavs[0]).shape[1]
    return 47

def run_fold_training():
    label_map = build_label_map(DATA_DIR)
    n_classes = len(label_map)
    id_to_label = {v: k for k, v in label_map.items()}
    time_dim = get_time_dim_example()

    folds = list_folds(DATA_DIR)  # fold1〜fold4 だけ

    # --- 評価スコア格納 ---
    all_accuracies = []
    all_macro_f1 = []
    all_weighted_f1 = []

    print(f"[Info] Using folds 1–4 for 4-fold cross-validation (fold5 excluded)")
    print(f"[Info] fold1–fold4 are rotated as eval fold")
    print(f"[Info] Input shape = ({N_MELS}, {time_dim}, 1)")

    # ============================================================
    # ★ ここが最重要ポイント
    # → val fold を置かない
    # → test fold = eval fold として扱う
    # ============================================================
    for i, eval_fold in enumerate(folds):
        print(f"\n[Fold {i + 1}]  Eval: {eval_fold.name}")

        # --- 1. Eval (Test) データ ---
        # これは「最終評価」のみに使う（学習中の監視には使わない）
        eval_paths, eval_labels_txt = list_wavs_and_labels(eval_fold)
        y_eval = np.array([label_map[t] for t in eval_labels_txt])

        # --- 2. Train データ収集 ---
        train_folds = [f for f in folds if f != eval_fold]
        full_train_paths, full_train_labels_txt = [], []
        for f in train_folds:
            p, l = list_wavs_and_labels(f)
            full_train_paths.extend(p)
            full_train_labels_txt.extend(l)

        full_train_y = np.array([label_map[t] for t in full_train_labels_txt])

        # --- 3. ★ Train をさらに Train / Validation に分割 ---
        # ここで VAL_SIZE を使用
        tr_paths, val_paths, tr_y, val_y = train_test_split(
            full_train_paths, full_train_y,
            test_size=VAL_SIZE,
            random_state=SEED,
            stratify=full_train_y  # クラスバランスを維持
        )

        # class_weights は tr_y (分割後の学習データ) に基づいて計算
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(n_classes),
            y=tr_y
        )
        class_weights = {i: w for i, w in enumerate(class_weights)}
        # Siren調整 (既存コード通り)
        for lbl, idx in label_map.items():
            if lbl.lower() == "siren":
                class_weights[idx] = class_weights[idx] * 0.3

        # --- Generators ---
        # tr_gen: 学習用 (Augmentationあり)
        tr_gen = MelSequence(tr_paths, tr_y.tolist(), BATCH_SIZE, n_classes, True)

        # val_gen: EarlyStopping監視用 (Augmentationなし, 評価対象は val_paths)
        val_gen = MelSequence(val_paths, val_y.tolist(), BATCH_SIZE, n_classes, False)

        # ev_gen: 最終テスト用 (Augmentationなし)
        ev_gen = MelSequence(eval_paths, y_eval.tolist(), BATCH_SIZE, n_classes, False)

        ##ここまで############

        # ---- Build model ----
        model = build_crnn(n_classes, time_dim)

        out_dir = Path(f"../../Result/runs_crnn_fold1to4_train_eval/{eval_fold.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Callbacks ----
        cbs = [
            EarlyStopping(monitor="val_accuracy", patience=PATIENCE,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(out_dir / "best.keras", monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=8, min_lr=1e-6, verbose=1)
        ]

        # ---- Train ----
        # --- Train ---
        # validation_data には val_gen を指定する
        history = model.fit(
            tr_gen,
            epochs=EPOCHS,
            validation_data=val_gen,  # ★ここを変更
            class_weight=class_weights,
            callbacks=cbs,
            verbose=1
        )

        # ---- Save Curves ----
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="eval_loss")
        plt.legend(); plt.grid(True)
        plt.title(f"Loss Curve ({eval_fold.name})")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="train_acc")
        plt.plot(history.history["val_accuracy"], label="eval_acc")
        plt.legend(); plt.grid(True)
        plt.title(f"Accuracy Curve ({eval_fold.name})")
        plt.tight_layout()
        plt.savefig(out_dir / "acc_curve.png")
        plt.close()

        # ---- Evaluate ----
        print(f"\n[Eval] Testing on {eval_fold.name}")
        y_pred_prob = model.predict(ev_gen, verbose=1)
        y_pred = np.argmax(y_pred_prob, axis=1)

        print(classification_report(
            y_eval, y_pred,
            target_names=[id_to_label[i] for i in range(n_classes)],
            digits=4))

        acc = accuracy_score(y_eval, y_pred)
        p,r,f1,_ = precision_recall_fscore_support(y_eval, y_pred, average="macro")
        _,_,f1w,_ = precision_recall_fscore_support(y_eval, y_pred, average="weighted")
        all_accuracies.append(acc)
        all_macro_f1.append(f1)
        all_weighted_f1.append(f1w)

        np.save(out_dir / "cm.npy", confusion_matrix(y_eval, y_pred))
        np.save(out_dir / "cm_norm.npy", confusion_matrix(y_eval, y_pred, normalize="true"))

        model.save(out_dir / "final.keras")

        # =========================================================
        # ★ ここに配置します
        # =========================================================
        # 1. バックエンドのグラフ（セッション）をクリア
        tf.keras.backend.clear_session()

        # 2. モデル変数を削除（Pythonの参照を切る）
        del model

        # 3. ガベージコレクションを強制実行（メモリをOSに返す）
        gc.collect()

        print(f"[Info] Fold {i + 1} finished. Memory cleared.")

    # ===== CV summary =====
    print("\n" + "="*70)
    print("[4-Fold Cross Validation Summary]")
    print("="*70)
    print(f"Mean Accuracy    : {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Mean Macro-F1    : {np.mean(all_macro_f1):.4f} ± {np.std(all_macro_f1):.4f}")
    print(f"Mean Weighted-F1 : {np.mean(all_weighted_f1):.4f} ± {np.std(all_weighted_f1):.4f}")
    print("="*70)


if __name__ == "__main__":
    run_fold_training()
