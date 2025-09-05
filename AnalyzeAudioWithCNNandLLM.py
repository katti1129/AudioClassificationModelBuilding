import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model# type: ignore
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D
from PIL import Image
import os
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. 設定 ---
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# CNNに入力する画像の固定サイズ
# (時間方向の長さはリサイズで固定する)
CNN_INPUT_SHAPE = (128, 256, 1)


# --- 2. 音声の前処理 ---
def preprocess_audio_to_image(audio_path):
    """音声ファイルを読み込み、CNNに入力できる画像形式のスペクトログラムに変換する"""
    try:
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        melspec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # スペクトログラムをリサイズ
        # PIL Imageに変換してリサイズ
        pil_img = Image.fromarray(log_melspec).resize((CNN_INPUT_SHAPE[1], CNN_INPUT_SHAPE[0]))
        img_array = np.array(pil_img)

        # チャンネル次元を追加
        img_array = np.expand_dims(img_array, axis=-1)

        return img_array

    except Exception as e:
        print(f"音声処理中にエラーが発生しました: {e}")
        return None


# --- 3. CNNによる特徴抽出モデルの構築 (Sequential) ---
def build_cnn_feature_extractor(input_shape):
    """SequentialモデルでCNN特徴抽出器を構築する"""
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # 最終的に1次元の特徴ベクトルに変換
        GlobalAveragePooling2D()
    ], name="cnn_feature_extractor")
    return model


# --- 4. LLMへの問い合わせ ---
def query_llm_with_audio_context(text_query, audio_features):
    """ユーザーの質問と音声の特徴を組み合わせてLLMに問い合わせる"""
    feature_str = ", ".join([f"{x:.4f}" for x in audio_features[:10]]) + ", ..."

    prompt = f"""
あなたは、音響分析とテキスト解釈の専門家AIです．
以下の2つの情報が与えられています．

1. **ユーザーからの質問**: "{text_query}"
2. **音声から抽出された特徴ベクトル**: [{feature_str}]

この音声特徴ベクトルは、ある音の音色やリズムなどの音響的な特性を凝縮したものです．
これらの情報を元に、ユーザーの質問に対して、専門家として自然な言葉で回答を生成してください．
"""

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"LLMとの通信中にエラーが発生しました: {e}"


# --- 5. メイン実行ブロック ---
if __name__ == "__main__":
    # --- セットアップ ---
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEYが設定されていません．.envファイルを確認してください．")
    genai.configure(api_key=api_key)

    # CNNモデルを構築
    cnn_feature_extractor = build_cnn_feature_extractor(CNN_INPUT_SHAPE)
    cnn_feature_extractor.summary()

    # --- 実行 ---
    # 分析したい音声ファイルのパス (踏切音のサンプル)
    audio_file_path = librosa.ex('brahms')
    print(f"\n分析対象の音声ファイル: {audio_file_path}")

    # ステップ1: 音声からスペクトログラム画像を作成
    spectrogram_image = preprocess_audio_to_image(audio_file_path)

    if spectrogram_image is not None:
        # ステップ2: CNNで特徴を抽出
        # バッチ次元を追加してモデルに入力
        input_for_cnn = np.expand_dims(spectrogram_image, axis=0)
        audio_features = cnn_feature_extractor.predict(input_for_cnn).flatten()
        print(f"音声から {len(audio_features)} 次元の特徴ベクトルを抽出しました．")

        # ステップ3: LLMに問い合わせ
        user_question = "この音声は何の音ですか？どのような特徴がありますか？"
        print(f"ユーザーの質問: 「{user_question}」")

        print("\nLLMに応答を生成させています．．．")
        llm_response = query_llm_with_audio_context(user_question, audio_features)

        print("\n--- LLMからの応答 ---")
        print(llm_response)