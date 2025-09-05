import librosa
from fastdtw import fastdtw
import numpy as np
import sounddevice as sd
import pickle


def record_audio(seconds=2, sr=16000):
    print("Recording...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()
    return np.squeeze(audio)

# 録音 → MFCC
audio = record_audio()
mfcc_test = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)

# テンプレート読み込み
with open('C:/Users/cpsla/PycharmProjects/LSTM/iwaizako/pkl/jishin_mfcc.pkl', "rb") as f:
    mfcc_template = pickle.load(f)

# 類似度比較
from scipy.spatial.distance import euclidean
dist, _ = fastdtw(mfcc_test.T, mfcc_template.T, dist=euclidean)

print("距離（類似度）:", dist)
