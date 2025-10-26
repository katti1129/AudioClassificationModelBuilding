# ==========================================================
# YAMNet特徴量 (1024次元) × RNNモデル
# 9分割交差検証 + 学習Loss/Accuracy可視化
# ==========================================================

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# type: ignore
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# --- 1. 設定 ---
FEATURES_DIR = Path('../features')  # 特徴量フォルダ
EPOCHS = 100
BATCH_SIZE = 32
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. 特徴量読み込み ---
print("YAMNet特徴量を読み込んでいます...")

with np.load(FEATURES_DIR / 'yamnet_features_train.npz', allow_pickle=True) as data:
    X_raw = data['features']
    y_raw = data['labels']
    CLASS_NAMES = data['class_names']

NUM_CLASSES = len(CLASS_NAMES)
print(f"クラス検出: {CLASS_NAMES}")
print(f"サンプル数: {len(X_raw)}")

# --- 3. パディング（時系列長を揃える） ---
print("シーケンス長を揃えています...")
max_len = max(len(x) for x in X_raw)
X = tf.keras.preprocessing.sequence.pad_sequences(X_raw, maxlen=max_len, padding='post', dtype='float32')
y = np.array(y_raw)
print(f"入力形状: {X.shape}, ラベル数: {y.shape}")

# --- 4. モデル構築関数 ---
def build_rnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Masking(mask_value=0.0),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(64)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 5. 9分割交差検証 ---
skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=SEED)
fold_results = []
y_true_all, y_pred_all = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    print(f"\n========== Fold {fold}/9 ==========")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_rnn_model((X.shape[1], X.shape[2]), NUM_CLASSES)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        ModelCheckpoint(f"yamnet_rnn_fold{fold:02d}.keras", monitor='val_loss', save_best_only=True)
    ]

    # --- 学習 ---
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0  # 結果を見やすくするため
    )

    # --- 学習曲線を表示 ---
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Fold {fold} の学習曲線", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, 'b-', label='訓練Loss')
    plt.plot(epochs_range, val_loss, 'r-', label='検証Loss')
    plt.xlabel('エポック')
    plt.ylabel('損失')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, 'b-', label='訓練Accuracy')
    plt.plot(epochs_range, val_acc, 'r-', label='検証Accuracy')
    plt.xlabel('エポック')
    plt.ylabel('正解率')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- 評価 ---
    y_pred_prob = model.predict(X_val)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1m = f1_score(y_val, y_pred, average='macro')

    print(f"[Fold {fold}] Accuracy={acc:.4f}, F1-macro={f1m:.4f}")

    fold_results.append({
        "fold": fold,
        "acc": acc,
        "f1": f1m,
        "train_loss": train_loss[-1],
        "val_loss": val_loss[-1]
    })
    y_true_all.append(y_val)
    y_pred_all.append(y_pred)

# --- 6. 総合結果 ---
y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

accs = np.array([r["acc"] for r in fold_results])
f1s = np.array([r["f1"] for r in fold_results])
val_losses = np.array([r["val_loss"] for r in fold_results])

print("\n===== 9-Fold Summary =====")
print(f"Accuracy : {accs.mean():.4f} ± {accs.std():.4f}")
print(f"F1-macro : {f1s.mean():.4f} ± {f1s.std():.4f}")
print(f"平均最終Loss : {val_losses.mean():.4f}")

# --- 7. 混同行列とレポート ---
print("\n分類レポート:")
print(classification_report(y_true_all, y_pred_all, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix (9-Fold Overall)')
plt.xlabel('予測ラベル (Predicted)')
plt.ylabel('正解ラベル (True)')
plt.tight_layout()
plt.show()
