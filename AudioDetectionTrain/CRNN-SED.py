# -*- coding: utf-8 -*-
import os, math, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# ====== 設定 ======
DATA_ROOT = Path("data/URBAN-SED")
AUDIO_DIR = DATA_ROOT / "audio"
ANNOT_PATH = DATA_ROOT / "annotations.csv"
CLASS_PATH = DATA_ROOT / "class_list.txt"

SR = 16000          # 16k 推奨（内部でリサンプリング）
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160    # 10ms (SR=16k -> 160)
WIN_LENGTH = 400    # 25ms

# 学習窓（リアルタイム想定）
WINDOW_SEC = 1.5
HOP_SEC = 0.5

BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-3
PATIENCE = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ====== ユーティリティ ======
def load_class_list(path):
    with open(path, "r") as f:
        classes = [l.strip() for l in f if l.strip()]
    idx = {c:i for i,c in enumerate(classes)}
    return classes, idx

CLASSES, CLASS2IDX = load_class_list(CLASS_PATH)
NUM_CLASSES = len(CLASSES)

def wav_to_logmel(y, sr=SR):
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=50, fmax=sr//2
    )
    logmel = librosa.power_to_db(S, ref=np.max)
    # 正規化（per-utterance）
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel.astype(np.float32)  # [n_mels, T]

def frame_times(T, hop_length=HOP_LENGTH, sr=SR):
    # 各フレームの時間（秒）
    return np.arange(T) * (hop_length / sr)

def events_to_frame_targets(events, frame_t, num_classes):
    # events: list of (onset, offset, class_idx)
    y = np.zeros((len(frame_t), num_classes), dtype=np.float32)
    for (on, off, cid) in events:
        m = (frame_t >= on) & (frame_t < off)
        y[m, cid] = 1.0
    return y  # [T, C]

def spec_augment(spec, freq_mask=8, time_mask=16, p=0.5):
    if np.random.rand() > p:
        return spec
    spec = spec.copy()
    # freq mask
    f = np.random.randint(0, freq_mask+1)
    f0 = np.random.randint(0, max(1, spec.shape[0]-f))
    spec[f0:f0+f, :] = 0
    # time mask
    t = np.random.randint(0, time_mask+1)
    t0 = np.random.randint(0, max(1, spec.shape[1]-t))
    spec[:, t0:t0+t] = 0
    return spec

# ====== データセット ======
class UrbanSEDWindowDataset(Dataset):
    def __init__(self, df_files, annot, audio_dir, window_sec=1.5, hop_sec=0.5, train=True):
        self.df_files = df_files.reset_index(drop=True)
        self.annot = annot
        self.audio_dir = audio_dir
        self.window = window_sec
        self.hop = hop_sec
        self.train = train

        # 事前に各ファイルのスライスインデックスを作る
        self.index = []  # (file, start_time, end_time, events_in_window)
        for _, r in self.df_files.iterrows():
            fn = r['file']
            dur = r['duration']
            events = self.annot[self.annot.file == fn]
            # 窓をスライド
            t = 0.0
            while t < max(1e-6, dur - 1e-6):
                t0 = t
                t1 = min(dur, t + self.window)
                # 窓内イベント抽出
                evs = []
                for _, e in events.iterrows():
                    # overlap check
                    if not (e['offset'] <= t0 or e['onset'] >= t1):
                        evs.append((max(e['onset'], t0)-t0, min(e['offset'], t1)-t0, CLASS2IDX[e['event_label']]))
                self.index.append((fn, t0, t1, evs))
                t += self.hop

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fn, t0, t1, evs = self.index[i]
        path = self.audio_dir / fn
        y, sr = librosa.load(path, sr=None, mono=True)
        # 切り出し
        s0 = int(t0 * sr); s1 = int(t1 * sr)
        y = y[s0:s1]
        # パディング
        target_len = int(self.window * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        # 特徴量
        mel = wav_to_logmel(y, sr=sr)            # [F, T]
        if self.train:
            mel = spec_augment(mel, freq_mask=8, time_mask=16, p=0.7)
        T = mel.shape[1]
        t_axis = frame_times(T)                  # [T]
        # フレームターゲット（窓内相対時刻→絶対時刻に変換不要。すでに窓開始0基準）
        events_abs = [(on, off, cid) for (on, off, cid) in evs]
        target = events_to_frame_targets(events_abs, t_axis, NUM_CLASSES)  # [T, C]
        # [C, F, T] 入力に整形（CNN用）
        x = torch.from_numpy(mel).unsqueeze(0).float()  # [1, F, T]
        return x, torch.from_numpy(target).float(), fn

# ====== モデル ======
class CRNN_SED(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=NUM_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        # 周波数次元を畳み込みで圧縮 → 時系列へ
        self.gru = nn.GRU(input_size=(n_mels//4)*128, hidden_size=256, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)  # フレームごとに C 次元
        )

    def forward(self, x):
        # x: [B, 1, F, T]
        h = self.cnn(x)                 # [B, 128, F/4, T/4]
        B, C, Fq, Tq = h.shape
        h = h.permute(0, 3, 1, 2).contiguous()   # [B, Tq, C, Fq]
        h = h.view(B, Tq, C*Fq)                   # [B, Tq, C*Fq]
        h, _ = self.gru(h)                        # [B, Tq, 512]
        out = self.fc(h)                          # [B, Tq, C]
        return out

# ====== データ読み込みと分割 ======
def load_metadata(annot_path, audio_dir):
    df = pd.read_csv(annot_path)
    # ファイル別の長さを取得
    files = sorted(df['file'].unique())
    rows = []
    for fn in files:
        path = audio_dir / fn
        if not path.exists():
            continue
        y, sr = librosa.load(path, sr=None, mono=True)
        dur = len(y)/sr
        rows.append({"file": fn, "duration": dur})
    df_files = pd.DataFrame(rows)
    return df, df_files

# ====== 評価（Segment-F1 @ 1s） ======
def segment_f1(y_true, y_pred, seg_frames, thr=0.5):
    """
    y_true, y_pred: list of [T, C] (各サンプルごと)
    seg_frames: 1秒相当のフレーム数 (SR/HOPから計算)
    """
    tp = fp = fn = 0
    for t_true, t_pred in zip(y_true, y_pred):
        # 1秒セグメントにまとめる（平均→閾値）
        T, C = t_true.shape
        nseg = int(math.ceil(T/seg_frames))
        for s in range(nseg):
            s0 = s*seg_frames
            s1 = min(T, (s+1)*seg_frames)
            gt = (t_true[s0:s1].mean(axis=0) >= 0.5).astype(np.int32)
            pr = (t_pred[s0:s1].mean(axis=0) >= thr).astype(np.int32)
            tp += int(np.logical_and(gt==1, pr==1).sum())
            fp += int(np.logical_and(gt==0, pr==1).sum())
            fn += int(np.logical_and(gt==1, pr==0).sum())
    precision = tp/(tp+fp+1e-8)
    recall    = tp/(tp+fn+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)
    return precision, recall, f1

# ====== 学習ループ ======
def train():
    annot, df_files = load_metadata(ANNOT_PATH, AUDIO_DIR)
    # クラス以外の行は削除（任意：クラス名を class_list.txt と揃える）
    annot = annot[annot["event_label"].isin(CLASSES)].copy()

    # ファイル単位で train/valid 分割（リーク防止）
    tr_files, va_files = train_test_split(df_files["file"].tolist(), test_size=0.1, random_state=SEED, shuffle=True)
    tr_df = df_files[df_files.file.isin(tr_files)]
    va_df = df_files[df_files.file.isin(va_files)]

    tr_ds = UrbanSEDWindowDataset(tr_df, annot, AUDIO_DIR, WINDOW_SEC, HOP_SEC, train=True)
    va_ds = UrbanSEDWindowDataset(va_df, annot, AUDIO_DIR, WINDOW_SEC, HOP_SEC, train=False)

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    model = CRNN_SED().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    best_f1, patience = 0.0, PATIENCE

    frames_per_sec = int(1.0 / (HOP_LENGTH / SR))  # 10ms hop → 100fps

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for x, y, _ in tqdm(tr_dl, desc=f"Train {epoch}"):
            x = x.to(DEVICE)               # [B, 1, F, T]
            y = y.to(DEVICE)               # [B, T, C]
            # 時間軸をモデル出力に合わせる（CNNで 1/4 に短縮 → 簡易に補間）
            out = model(x)                 # [B, Tq, C]
            # y を Tq に縮約（平均プーリング）
            B, Tq, C = out.shape
            T = y.shape[1]
            ratio = T / Tq
            yq = []
            for i in range(Tq):
                s0 = int(i*ratio); s1 = int(min(T, (i+1)*ratio))
                yq.append(y[:, s0:s1, :].mean(dim=1))
            yq = torch.stack(yq, dim=1)   # [B, Tq, C]

            loss = crit(out, yq)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr_dl.dataset)

        # === validation ===
        model.eval()
        va_loss = 0.0
        ys_true, ys_pred = [], []
        with torch.no_grad():
            for x, y, _ in tqdm(va_dl, desc=f"Valid {epoch}"):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                out = model(x)             # [B, Tq, C]
                # y を Tq に縮約
                B, Tq, C = out.shape
                T = y.shape[1]
                ratio = T / Tq
                yq = []
                for i in range(Tq):
                    s0 = int(i*ratio); s1 = int(min(T, (i+1)*ratio))
                    yq.append(y[:, s0:s1, :].mean(dim=1))
                yq = torch.stack(yq, dim=1)
                loss = crit(out, yq)
                va_loss += loss.item() * x.size(0)

                # 推論確率をシグモイドで取得 → CPU numpy
                prob = torch.sigmoid(out).cpu().numpy()      # [B, Tq, C]
                yq_np = yq.cpu().numpy()                     # [B, Tq, C]
                # 1秒セグメント F1 用に、元の fps（約100fps）に合わせて補間（簡易）
                # ここでは Tq をそのまま fps=~25 として扱い、セグメント幅を Tq/ (sec_len) にする簡便法
                ys_pred += [p for p in prob]
                ys_true += [t for t in yq_np]

        va_loss /= len(va_dl.dataset)
        # 簡便に Tq ≈ 1/4 のfps → 1秒相当のセグメントは WINDOW_SEC/HOP_SEC と独立に、ここでは Tq/ (WINDOW_SEC) を fps と見做す
        # より厳密には前段の縮約比から fps を算出してください
        # ここでは 1秒セグメント= max(1, int(Tq / WINDOW_SEC)) フレームとします
        if len(ys_true) > 0:
            Tq_est = ys_true[0].shape[0]
            seg_frames = max(1, int(Tq_est / WINDOW_SEC))
            p, r, f1 = segment_f1([t for t in ys_true], [p for p in ys_pred], seg_frames, thr=0.5)
        else:
            p = r = f1 = 0.0

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f}  valid_loss={va_loss:.4f}  F1@1s={f1:.4f} (P={p:.3f}, R={r:.3f})")

        # EarlyStopping + Best保存
        if f1 > best_f1:
            best_f1 = f1
            patience = PATIENCE
            torch.save({"model": model.state_dict(),
                        "classes": CLASSES,
                        "cfg": dict(SR=SR,N_MELS=N_MELS,N_FFT=N_FFT,HOP_LENGTH=HOP_LENGTH,WIN_LENGTH=WIN_LENGTH)},
                        "best_crnn_sed_urban.pt")
            print(f"  -> Saved best (F1={best_f1:.4f})")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.")
                break

# ====== 学習ループ ======
def train():
    annot, df_files = load_metadata(ANNOT_PATH, AUDIO_DIR)
    annot = annot[annot["event_label"].isin(CLASSES)].copy()

    tr_files, va_files = train_test_split(df_files["file"].tolist(), test_size=0.1, random_state=SEED, shuffle=True)
    tr_df = df_files[df_files.file.isin(tr_files)]
    va_df = df_files[df_files.file.isin(va_files)]

    tr_ds = UrbanSEDWindowDataset(tr_df, annot, AUDIO_DIR, WINDOW_SEC, HOP_SEC, train=True)
    va_ds = UrbanSEDWindowDataset(va_df, annot, AUDIO_DIR, WINDOW_SEC, HOP_SEC, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = CRNN_SED().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    best_f1, patience = 0.0, PATIENCE
    train_losses, valid_losses = [], []
    frames_per_sec = int(1.0 / (HOP_LENGTH / SR))

    # === ログをファイルに同時出力 ===
    log_path = Path("training_log.txt")
    sys.stdout = open(log_path, "w", buffering=1)

    print("===== TRAINING START =====")
    print(f"Device: {DEVICE}, Classes: {NUM_CLASSES}")
    print(f"Train samples: {len(tr_ds)}, Valid samples: {len(va_ds)}\n")

    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for x, y, _ in tqdm(tr_dl, desc=f"Train {epoch}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            B, Tq, C = out.shape
            T = y.shape[1]
            ratio = T / Tq
            yq = []
            for i in range(Tq):
                s0 = int(i*ratio); s1 = int(min(T, (i+1)*ratio))
                yq.append(y[:, s0:s1, :].mean(dim=1))
            yq = torch.stack(yq, dim=1)
            loss = crit(out, yq)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr_dl.dataset)

        # === validation ===
        model.eval()
        va_loss = 0.0
        ys_true, ys_pred = [], []
        with torch.no_grad():
            for x, y, _ in tqdm(va_dl, desc=f"Valid {epoch}"):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                out = model(x)
                B, Tq, C = out.shape
                T = y.shape[1]
                ratio = T / Tq
                yq = []
                for i in range(Tq):
                    s0 = int(i*ratio); s1 = int(min(T, (i+1)*ratio))
                    yq.append(y[:, s0:s1, :].mean(dim=1))
                yq = torch.stack(yq, dim=1)
                loss = crit(out, yq)
                va_loss += loss.item() * x.size(0)
                prob = torch.sigmoid(out).cpu().numpy()
                yq_np = yq.cpu().numpy()
                ys_pred += [p for p in prob]
                ys_true += [t for t in yq_np]
        va_loss /= len(va_dl.dataset)

        # === F1計算 ===
        if len(ys_true) > 0:
            Tq_est = ys_true[0].shape[0]
            seg_frames = max(1, int(Tq_est / WINDOW_SEC))
            p, r, f1 = segment_f1(ys_true, ys_pred, seg_frames, thr=0.5)
        else:
            p = r = f1 = 0.0

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f}  valid_loss={va_loss:.4f}  F1@1s={f1:.4f} (P={p:.3f}, R={r:.3f})")

        train_losses.append(tr_loss)
        valid_losses.append(va_loss)

        # EarlyStopping + Best保存
        if f1 > best_f1:
            best_f1 = f1
            patience = PATIENCE
            torch.save({
                "model": model.state_dict(),
                "classes": CLASSES,
                "cfg": dict(SR=SR, N_MELS=N_MELS, N_FFT=N_FFT, HOP_LENGTH=HOP_LENGTH)
            }, "best_crnn_sed_urban.pt")
            print(f"  -> Saved best (F1={best_f1:.4f})")
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break

    # === 終了処理 ===
    sys.stdout.close()
    sys.stdout = sys.__stdout__  # 標準出力を戻す
    print(f"ログを保存しました: {log_path.resolve()}")

    # === ロス曲線をプロット ===
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.title("Loss per Epoch (Overfitting Check)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    print("ロス曲線を保存しました: loss_curve.png")

if __name__ == "__main__":
    train()
