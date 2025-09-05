# 03_train_yamnet_rnn.py

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model# type: ignore
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint# type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences# type: ignore
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # 日本語表示のため

# --- 1. 設定・パラメータ ---
FEATURES_DIR = Path('../features')
MODEL_SAVE_PATH = 'best_yamnet_rnn_model.keras'
BATCH_SIZE = 32
EPOCHS = 100

# --- 2. データ準備 ---
print("抽出済みの特徴量を読み込んでいます．．．")

try:
    # 訓練・検証・テストの各データをロード
    with np.load(FEATURES_DIR / 'yamnet_features_train.npz', allow_pickle=True) as data:
        X_train_raw = data['features']
        y_train_labels = data['labels']
        CLASS_NAMES = data['class_names']

    with np.load(FEATURES_DIR / 'yamnet_features_validation.npz', allow_pickle=True) as data:
        X_val_raw = data['features']
        y_val_labels = data['labels']

    with np.load(FEATURES_DIR / 'yamnet_features_test.npz', allow_pickle=True) as data:
        X_test_raw = data['features']
        y_test_labels = data['labels']
except FileNotFoundError as e:
    print(f"エラー: {e}")
    print("特徴量ファイルが見つかりません．先に`02_extract_yamnet_features.py`を実行してください．")
    exit()

NUM_CLASSES = len(CLASS_NAMES)

# パディングでシーケンスの長さを揃える
# データセット全体の最大長に合わせる
print("データのパディング処理を行っています．．．")
max_len = max(len(x) for x in np.concatenate([X_train_raw, X_val_raw, X_test_raw]))
print(f"シーケンスの最大長: {max_len}")

X_train = pad_sequences(X_train_raw, maxlen=max_len, padding='post', dtype='float32')
X_val = pad_sequences(X_val_raw, maxlen=max_len, padding='post', dtype='float32')
X_test = pad_sequences(X_test_raw, maxlen=max_len, padding='post', dtype='float32')

# ラベルをカテゴリカル形式に変換
y_train = tf.keras.utils.to_categorical(y_train_labels, num_classes=NUM_CLASSES)
y_val = tf.keras.utils.to_categorical(y_val_labels, num_classes=NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test_labels, num_classes=NUM_CLASSES)

print(f"訓練: {len(X_train)}件, 検証: {len(X_val)}件, テスト: {len(X_test)}件")

# --- 3. RNNモデルの構築 ---
print("RNNモデルを構築しています．．．")
input_shape = (X_train.shape[1], X_train.shape[2])  # (時間ステップ数, 特徴次元数)
model = Sequential([
    Input(shape=input_shape),
    Masking(mask_value=0.0),  # パディングした部分を無視する
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
model.summary()

# --- 4. モデルのコンパイルと学習 ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
             ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)]

print("モデルの学習を開始します．．．")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

# --- 5. 学習曲線のプロット ---
print("\n学習曲線をプロットします．．．")
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='訓練データ (Train)')
plt.plot(epochs, val_acc, 'ro-', label='検証データ (Val)')
plt.title('訓練・検証データの正解率')
plt.xlabel('エポック数')
plt.ylabel('正解率 (Accuracy)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='訓練データ (Train)')
plt.plot(epochs, val_loss, 'ro-', label='検証データ (Val)')
plt.title('訓練・検証データの損失')
plt.xlabel('エポック数')
plt.ylabel('損失 (Loss)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 6. テストデータでの最終評価 ---
print(f"\n最適なモデル({MODEL_SAVE_PATH})でテストデータに対する最終評価を行います．．．")
best_model = load_model(MODEL_SAVE_PATH)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# 混同行列と分類レポート
print("\n混同行列と分類レポートを作成します．．．")
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('テストデータに対する混同行列 (Confusion Matrix on Test Data)')
plt.xlabel('予測ラベル (Predicted Label)')
plt.ylabel('正解ラベル (True Label)')
plt.show()

print("\n分類レポート (Test Data):")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))