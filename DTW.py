import librosa
import pickle
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# --- 1. テンプレート（pklファイル）を読み込み ---
template_path = 'C:/Users/cpsla/PycharmProjects/LSTM/iwaizako/pkl/jishin_mfcc.pkl'  # 自分のパスに変更
with open(template_path, 'rb') as f:
    mfcc_template = pickle.load(f)

# --- 2. 比較対象のwavファイルを読み込み ---
wav_path = 'C:/Users/cpsla/PycharmProjects/LSTM/iwaizako/audio/misairu-sokuho.wavone '  # 比較したい音
y, sr = librosa.load(wav_path, sr=None)

# --- 3. wavファイルをMFCCに変換 ---
mfcc_test = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# --- 4. DTWによる類似度（距離）の計算 ---
# fastdtwでは次元の順番を (時間, 特徴量) に揃える必要があるので `.T` します
distance, _ = fastdtw(mfcc_template.T, mfcc_test.T, dist=euclidean)

# --- 5. 結果表示 ---
print("✅ DTW距離（類似度）:", distance)
