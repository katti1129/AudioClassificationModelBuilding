#Foldをfor文でまわす．クラスごとにfor文で回す．1.0sでスライドウィンドウ方式適用．CreateLeakfreeSplit_5Fold.pyの後に使用．
import librosa
import soundfile as sf
from pathlib import Path

SRC = Path("../../data/clean_folds_5fold")
OUT = Path("../../data/clean_folds_5fold_1p")

CLASSES = ["siren", "other", "silence"]

SR = 16000
WIN = 1.0
HOP = 0.5
WIN_SAMPLES = int(SR * WIN)
HOP_SAMPLES = int(SR * HOP)

def main():
    for fold in sorted(SRC.glob("fold*")):
        for cls in CLASSES:
            out_dir = OUT / fold.name / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            files = sorted((fold / cls).glob("*.wav"))
            for wav in files:
                y, sr = librosa.load(wav, sr=SR)
                length = len(y)
                idx, wid = 0, 0

                while idx + WIN_SAMPLES <= length:
                    chunk = y[idx: idx + WIN_SAMPLES]
                    out_path = out_dir / f"{wav.stem}_win{wid}.wav"
                    sf.write(out_path, chunk, SR)
                    wid += 1
                    idx += HOP_SAMPLES

                print(f"{fold.name} - {cls} - {wav.name}: {wid} windows 生成")

    print("完了：全 fold に対して 1.5秒スライドウィンドウ生成")

if __name__ == "__main__":
    main()
