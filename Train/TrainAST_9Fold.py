# -*- coding: utf-8 -*-
# save as: TrainAST_9Fold.py

import os
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, get_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import seaborn as sns
import random

# =========================================================
# 1) 設定
# =========================================================
DATA_DIR = Path("../../data/UrbanSound8K_split_4sec")
MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("../../Result/runs_ast_fold1to9", exist_ok=True)
print(f"Using device: {DEVICE}")

# =========================================================
# 2) ユーティリティ
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

def list_folds(data_dir: Path):
    return [p for p in sorted(data_dir.iterdir())
            if p.is_dir() and p.name.startswith("fold")
            and not p.name.endswith("10")]

def list_wavs_and_labels(fold_dir: Path):
    wavs, labels = [], []
    for cls_dir in sorted(fold_dir.iterdir()):
        if cls_dir.is_dir():
            for wav in cls_dir.glob("*.wav"):
                wavs.append(str(wav))
                labels.append(cls_dir.name)
    return wavs, labels

def build_label_map(data_dir: Path):
    classes = sorted({cls.name for f in data_dir.iterdir()
                      if f.is_dir() and not f.name.endswith("10")
                      for cls in f.iterdir() if cls.is_dir()})
    return {c: i for i, c in enumerate(classes)}

# =========================================================
# 3) データセット
# =========================================================
class AudioDataset(Dataset):
    def __init__(self, paths, labels, feature_extractor):
        self.paths = paths
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        wav, _ = librosa.load(path, sr=self.feature_extractor.sampling_rate, mono=True)
        return {"raw": wav, "label": label}

def collate_fn(batch, feature_extractor):
    raw_audios = [item["raw"] for item in batch]
    labels = [item["label"] for item in batch]
    inputs = feature_extractor(raw_audios, sampling_rate=feature_extractor.sampling_rate,
                               return_tensors="pt", padding=True)
    return {"input_values": inputs.input_values, "labels": torch.tensor(labels)}

# =========================================================
# 4) 学習関数
# =========================================================
def train_one_fold(train_loader, val_loader, model, optimizer, scheduler, n_epochs, device, out_dir):
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        progress = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{n_epochs}]")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                total_val_loss += loss.item()
                total += batch["labels"].size(0)
                correct += (preds == batch["labels"]).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f}")

    # --- plot learning curves ---
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend(); plt.grid(True)
    plt.title("Loss Curve")
    plt.savefig(out_dir / "loss_curve.png"); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend(); plt.grid(True)
    plt.title("Validation Accuracy")
    plt.savefig(out_dir / "acc_curve.png"); plt.close()

    return model, history

# =========================================================
# 5) クロスバリデーション
# =========================================================
def run_fold_training():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
    label_map = build_label_map(DATA_DIR)
    id_to_label = {v: k for k, v in label_map.items()}
    n_classes = len(label_map)
    folds = list_folds(DATA_DIR)

    all_acc, all_f1_macro, all_f1_weighted = [], [], []

    print(f"\n[Info] Start 9-Fold CV ({n_classes} classes)")
    for i, test_fold in enumerate(folds):
        val_fold = folds[(i + 1) % len(folds)]
        train_folds = [f for f in folds if f not in (test_fold, val_fold)]

        print(f"\n========== Fold {i+1} ==========")
        print(f"Test: {test_fold.name} | Val: {val_fold.name}")

        # --- data split ---
        train_paths, train_labels = [], []
        for f in train_folds:
            p, l = list_wavs_and_labels(f)
            train_paths.extend(p)
            train_labels.extend([label_map[t] for t in l])

        val_paths, val_labels = list_wavs_and_labels(val_fold)
        val_labels = [label_map[t] for t in val_labels]

        test_paths, test_labels = list_wavs_and_labels(test_fold)
        test_labels = [label_map[t] for t in test_labels]

        # --- dataloader ---
        train_ds = AudioDataset(train_paths, train_labels, feature_extractor)
        val_ds = AudioDataset(val_paths, val_labels, feature_extractor)
        test_ds = AudioDataset(test_paths, test_labels, feature_extractor)

        collate = lambda x: collate_fn(x, feature_extractor)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate)

        # --- model ---
        model = AutoModelForAudioClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=n_classes,
            label2id=label_map,
            id2label=id_to_label,
            ignore_mismatched_sizes=True,
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = NUM_EPOCHS * len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        out_dir = Path(f"../../Result/runs_ast_fold1to9/{test_fold.name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- train ---
        model, _ = train_one_fold(train_loader, val_loader, model, optimizer, scheduler,
                                  NUM_EPOCHS, DEVICE, out_dir)

        # --- eval ---
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                pred = torch.argmax(outputs.logits, dim=-1)
                preds.extend(pred.cpu().numpy())
                trues.extend(batch["labels"].cpu().numpy())

        acc = accuracy_score(trues, preds)
        p, r, f1, _ = precision_recall_fscore_support(trues, preds, average="macro")
        _, _, f1w, _ = precision_recall_fscore_support(trues, preds, average="weighted")
        all_acc.append(acc)
        all_f1_macro.append(f1)
        all_f1_weighted.append(f1w)

        print(classification_report(trues, preds, target_names=[id_to_label[i] for i in range(n_classes)], digits=4))
        print(f"[Fold {i+1}] acc={acc:.4f}, macro-F1={f1:.4f}, weighted-F1={f1w:.4f}")

        # --- confusion matrix ---
        cm = confusion_matrix(trues, preds, normalize="true")
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=[id_to_label[i] for i in range(n_classes)],
                    yticklabels=[id_to_label[i] for i in range(n_classes)])
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title(f"Confusion Matrix - {test_fold.name}")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png"); plt.close()

        # --- save model ---
        model.save_pretrained(out_dir / "model")
        feature_extractor.save_pretrained(out_dir / "model")

    # =========================================================
    # 6) クロスバリデーション平均
    # =========================================================
    print("\n========== Cross Validation Summary ==========")
    print(f"Mean Accuracy    : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean Macro-F1    : {np.mean(all_f1_macro):.4f} ± {np.std(all_f1_macro):.4f}")
    print(f"Mean Weighted-F1 : {np.mean(all_f1_weighted):.4f} ± {np.std(all_f1_weighted):.4f}")

# =========================================================
# 7) 実行
# =========================================================
if __name__ == "__main__":
    run_fold_training()
