#音源をmfccに変換

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# --- 1. wavファイルの読み込み ---
file_path = 'C:/Users/cpsla/PycharmProjects/LSTM/iwaizako/audio/jishin-sokuho.wav'  # ←ここに自分のwavファイルのパスを入れる
y, sr = librosa.load(file_path, sr=None)  # sr=Noneで元のサンプリングレートを維持

# --- 2. MFCCの抽出 ---
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 13次元のMFCC（一般的）

# --- 3. MFCCの表示 ---
save_path = 'C:/Users/cpsla/PycharmProjects/LSTM/iwaizako/pkl/jishin_mfcc.pkl'  # 任意の保存場所
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # フォルダがなければ作成

with open(save_path, 'wb') as f:
    pickle.dump(mfccs, f)
