import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, get_scheduler
from torch.optim import AdamW
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import json

# --- 1. 設定 ---
DATA_DIR = Path('../../data/ESC-50-master/organized/segmented_data_split')
MODEL_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5

# GPUが利用可能か確認し、デバイスを設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用するデバイス: {device}")

# --- 2. データセットの準備 ---
train_dir = DATA_DIR / 'train'
CLASS_NAMES = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
print(f"{NUM_CLASSES}個のクラスを検出しました: {', '.join(CLASS_NAMES)}")


def load_audio_paths_and_labels(data_dir):
    paths, labels = [], []
    for class_name, label_id in label2id.items():
        class_dir = Path(data_dir) / class_name
        for file_path in class_dir.glob('*.wav'):
            paths.append(str(file_path))
            labels.append(label_id)
    return paths, labels


train_paths, train_labels = load_audio_paths_and_labels(DATA_DIR / 'train')
val_paths, val_labels = load_audio_paths_and_labels(DATA_DIR / 'validation')
test_paths, test_labels = load_audio_paths_and_labels(DATA_DIR / 'test')

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)


class AudioDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        wav, _ = librosa.load(path, sr=feature_extractor.sampling_rate, mono=True)
        return {"raw": wav, "label": label}


def collate_fn(batch):
    raw_audios = [item["raw"] for item in batch]
    labels = [item["label"] for item in batch]
    inputs = feature_extractor(raw_audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt",
                               padding=True)
    return {"input_values": inputs.input_values, "labels": torch.tensor(labels)}


train_dataset = AudioDataset(train_paths, train_labels)
val_dataset = AudioDataset(val_paths, val_labels)
test_dataset = AudioDataset(test_paths, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- 3. モデルの構築と学習 ---
print("\n事前学習済みASTモデルをロードしています...")
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_CLASSES,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
).to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

print("\nモデルのファインチューニングを開始します...")
history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_dataloader)
    history['train_loss'].append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            total += batch["labels"].size(0)
            correct += (predictions == batch["labels"]).sum().item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct / total
    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(val_accuracy)

    print(
        f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# --- 4. 評価 ---
print("\nテストデータでモデルを評価します...")
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        test_total += batch["labels"].size(0)
        test_correct += (predictions == batch["labels"]).sum().item()

test_accuracy = test_correct / test_total
print(f'\nTest accuracy: {test_accuracy:.4f}')


# --- 5. 学習履歴のグラフ化 ---
def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(history['train_loss'])))
    plt.legend()
    plt.grid(True)
    plt.show()


print("\n学習履歴のグラフを表示します...")
plot_history(history)

# --- 6. モデルの保存 ---
model.save_pretrained("./ast_finetuned_model_pytorch")
feature_extractor.save_pretrained("./ast_finetuned_model_pytorch")
print("\nファインチューニング済みモデルを ./ast_finetuned_model_pytorch に保存しました。")
