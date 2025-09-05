import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- 1. 設定 ---
# ご自身のWAVファイルパスに書き換えてください
# file_path = 'your_audio.wav'
# librosaのサンプル音声を使用
file_path = './data/4-PoliceCar/audiostock_998809.wav'

# MFCC計算用のパラメータ
SAMPLE_RATE = 22050 # サンプリングレート
N_FFT = 2048      # FFTのポイント数
HOP_LENGTH = 512    # フレームのシフト長
N_MELS = 128        # メルフィルターバンクの数
N_MFCC = 13         # MFCCの次元数

# --- 2. 音声ファイルの読み込み ---
y, sr = librosa.load(file_path, sr=SAMPLE_RATE)


# --- 3. 各過程の計算とグラフ描画 ---

# グラフ1: 元の音声波形
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('1. Original Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ1: プリエンファシス
y_preemphasized = librosa.effects.preemphasis(y)

# グラフ2: プリエンファシス後の波形
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y_preemphasized, sr=sr)
plt.title('2. Waveform after Pre-emphasis')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ2,3: フレーミングと窓関数
# 中央付近の1フレームを例として取り出す
frame_sample = y_preemphasized[10000:10000+N_FFT]
frame_windowed = frame_sample * np.hamming(N_FFT)

# グラフ3: 1フレームを切り出した後の波形（窓関数適用後）
plt.figure(figsize=(14, 5))
plt.plot(frame_windowed)
plt.title('3. A Single Frame after Windowing')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ4: 短時間フーリエ変換(STFT)とパワースペクトル
# 1フレーム分のSTFT
stft_frame = np.abs(librosa.stft(frame_windowed, n_fft=N_FFT))**2

# グラフ4: パワースペクトル（1フレーム分）
plt.figure(figsize=(14, 5))
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
plt.plot(freqs, stft_frame)
plt.title('4. Power Spectrum (of a single frame)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ5: メルスペクトログラム
# 音声全体で計算
melspec = librosa.feature.melspectrogram(y=y_preemphasized, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)

# グラフ5: メルスペクトログラム
plt.figure(figsize=(10, 4))
librosa.display.specshow(melspec, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f')
plt.title('5. Mel Spectrogram')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ6: 対数化（ログメルスペクトログラム）
log_melspec = librosa.power_to_db(melspec, ref=np.max)

# グラフ6: ログメルスペクトログラム
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_melspec, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('6. Log-Mel Spectrogram')
plt.tight_layout()
plt.show()

# ----------------------------------------------------

# ステップ7,8: 離散コサイン変換(DCT)と係数選択（MFCC）
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, n_mfcc=N_MFCC)

# グラフ7: MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('7. Mel-Frequency Cepstral Coefficients (MFCCs)')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()