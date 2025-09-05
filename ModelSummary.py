from tensorflow.keras.models import load_model# type: ignore
import numpy as np

# 学習済みモデルをロード
model = load_model("C:/Users/cpsla/PycharmProjects/LSTM/試し/lstm_model_5s3f.h5")

# モデルのサマリーを表示 (これは問題なく動作します)
model.summary()

# モデルのレイヤー情報を表示
print("\nLayer details:")
for layer in model.layers:
    # layer.output_shape を使用
    print(layer.name, layer.output_shape)

# その他の処理を続ける

# モデルの重みを確認（任意のレイヤーを選択）
# 例: 最初のLSTMレイヤーの重み
# if 'lstm' in model.layers[0].name:
#     print(model.layers[0].get_weights())