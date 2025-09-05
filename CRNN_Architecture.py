import tensorflow as tf
from tensorflow.keras.models import Sequential# type: ignore
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Activation, Reshape, Bidirectional, LSTM, Dense
)
from tensorflow.keras.utils import plot_model


def build_crnn_model(input_shape, num_classes):
    """
    CRNNモデルを構築する関数
    """
    model = Sequential([
        Input(shape=input_shape, name='input_spectrogram'),

        # --- CNN Block ---
        Conv2D(32, (3, 3), padding='same', name='conv1'),
        BatchNormalization(name='bn1'),
        Activation('relu', name='relu1'),
        MaxPooling2D(pool_size=(2, 2), name='pool1'),

        Conv2D(64, (3, 3), padding='same', name='conv2'),
        BatchNormalization(name='bn2'),
        Activation('relu', name='relu2'),
        MaxPooling2D(pool_size=(2, 2), name='pool2'),

        Conv2D(128, (3, 3), padding='same', name='conv3'),
        BatchNormalization(name='bn3'),
        Activation('relu', name='relu3'),
        MaxPooling2D(pool_size=(2, 2), name='pool3'),

        # --- Bridge (Reshape) ---
        Reshape((-1, 128 * (input_shape[0] // 8)), name='reshape_to_sequence'),

        # --- RNN Block ---
        Bidirectional(LSTM(64, return_sequences=True), name='bilstm1'),
        Bidirectional(LSTM(64), name='bilstm2'),

        # --- Classifier Block ---
        Dense(64, activation='relu', name='dense1'),
        Dense(num_classes, activation='softmax', name='output_softmax')
    ])
    return model


# --- モデルの構造を画像化 ---

# モデルに入力するデータの形状を仮定
# (周波数ビン, 時間ステップ数, チャンネル数)
INPUT_SHAPE = (128, 157, 1)
NUM_CLASSES = 7

# モデルを構築
crnn_model = build_crnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

# モデルの構造をPNGファイルとして保存
try:
    plot_model(
        crnn_model,
        to_file='crnn_model_architecture.png',
        show_shapes=True,  # 各層の入出力の形状を表示
        show_layer_names=True,  # 各層の名前を表示
        show_dtype=False,  # データ型を非表示
        show_layer_activations=True  # 各層の活性化関数を表示
    )
    print("モデルの構造図を 'crnn_model_architecture.png' として保存しました。")
except ImportError as e:
    print(f"エラー: {e}")
    print("画像を生成するには、pydotとGraphvizが必要です。")
    print("conda install python-graphviz または pip install pydot を実行してください。")