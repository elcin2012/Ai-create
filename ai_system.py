from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

CHECKPOINT_PATH = Path("checkpoint.pt")
META_PATH = Path("meta.json")


@dataclass
class Config:
    max_len: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 25
    lr: float = 1e-3


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zа-я0-9\s?!.,]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def build_toy_dataset() -> List[Tuple[str, str]]:
    # intent labels: greeting, order_status, refund, goodbye
    return [
        ("Привет", "greeting"),
        ("Здравствуйте", "greeting"),
        ("Добрый день", "greeting"),
        ("Хай", "greeting"),
        ("Где мой заказ", "order_status"),
        ("Когда приедет посылка", "order_status"),
        ("Хочу узнать статус доставки", "order_status"),
        ("Проверьте мой трек номер", "order_status"),
        ("Хочу вернуть товар", "refund"),
        ("Оформить возврат", "refund"),
        ("Верните деньги", "refund"),
        ("Как сделать возврат", "refund"),
        ("Спасибо до свидания", "goodbye"),
        ("Пока", "goodbye"),
        ("Всего доброго", "goodbye"),
        ("До встречи", "goodbye"),
    ]


def build_vocab(samples: List[Tuple[str, str]]) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for text, _ in samples:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def build_label_maps(samples: List[Tuple[str, str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({label for _, label in samples})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens][:max_len]
    mask = [1] * len(ids)

    while len(ids) < max_len:
        ids.append(vocab[PAD_TOKEN])
        mask.append(0)

    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class IntentDataset(Dataset):
    def __init__(self, samples, vocab, label2id, max_len: int):
        self.samples = samples
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        input_ids, attention_mask = encode_text(text, self.vocab, self.max_len)
        y = torch.tensor(self.label2id[label], dtype=torch.long)
        return input_ids, attention_mask, y


class IntentTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_labels: int, cfg: Config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.embed_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        # True means "ignore" for src_key_padding_mask, so invert attention mask.
        padding_mask = ~attention_mask
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        mask = attention_mask.unsqueeze(-1)
        x = x * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def train_model(cfg: Config) -> None:
    samples = build_toy_dataset()
    random.shuffle(samples)

    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    vocab = build_vocab(samples)
    label2id, id2label = build_label_maps(samples)

    train_ds = IntentDataset(train_samples, vocab, label2id, cfg.max_len)
    val_ds = IntentDataset(val_samples, vocab, label2id, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = IntentTransformer(vocab_size=len(vocab), num_labels=len(label2id), cfg=cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0

        for input_ids, attention_mask, y in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for input_ids, attention_mask, y in val_loader:
                logits = model(input_ids, attention_mask)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / max(1, total)
        avg_loss = train_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        best_state = model.state_dict()

    torch.save(best_state, CHECKPOINT_PATH)
    meta = {
        "vocab": vocab,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "config": cfg.__dict__,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model to {CHECKPOINT_PATH} and metadata to {META_PATH}")


def load_model() -> Tuple[IntentTransformer, Dict[str, int], Dict[int, str], Config]:
    if not CHECKPOINT_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run: python ai_system.py train")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    cfg = Config(**meta["config"])

    model = IntentTransformer(vocab_size=len(vocab), num_labels=len(label2id), cfg=cfg)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, vocab, id2label, cfg


def predict(model: IntentTransformer, vocab: Dict[str, int], id2label: Dict[int, str], cfg: Config, text: str):
    input_ids, attention_mask = encode_text(text, vocab, cfg.max_len)
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())
    return {
        "intent": id2label[pred_id],
        "confidence": float(probs[pred_id].item()),
    }


app = FastAPI(title="Intent Classifier API")


class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    model, vocab, id2label, cfg = load_model()
    return predict(model, vocab, id2label, cfg, req.text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_system.py [train|serve|predict]")
        return

    command = sys.argv[1].lower()

    if command == "train":
        cfg = Config()
        train_model(cfg)
    elif command == "serve":
        import uvicorn

        uvicorn.run("ai_system:app", host="0.0.0.0", port=8000, reload=False)
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python ai_system.py predict 'your text'")
            return
        text = sys.argv[2]
        model, vocab, id2label, cfg = load_model()
        print(predict(model, vocab, id2label, cfg, text))
    else:
        print("Unknown command. Use: train | serve | predict")


if __name__ == "__main__":
    main()
