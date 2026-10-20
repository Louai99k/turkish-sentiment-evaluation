"""
Docker-first Stage 2 non-transformer baseline runner.

Runs:
- BiLSTM sequence baseline
- fastText Turkish sentence embeddings + Logistic Regression

Artifacts are saved under outputs/experiments/<exp_id>/.
"""
import argparse
import gzip
import json
import os
import pickle
import random
import sys
from collections import Counter
from datetime import datetime
from urllib.error import ContentTooShortError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import calculate_class_weights, load_winvoker_dataset, preprocess_dataset
from src.evaluate import plot_confusion_matrix, plot_training_history


LABEL_MAPPING = {0: "Positive", 1: "Negative", 2: "Notr"}
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def tokenize_text(text):
    return str(text).split()


def build_vocab(texts, min_freq=2, max_vocab_size=100000):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common(max_vocab_size - len(vocab)):
        if count < min_freq:
            break
        vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_length):
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize_text(text)[:max_length]]
    attention_mask = [1] * len(token_ids)

    if len(token_ids) < max_length:
        pad_len = max_length - len(token_ids)
        token_ids.extend([vocab[PAD_TOKEN]] * pad_len)
        attention_mask.extend([0] * pad_len)

    return token_ids, attention_mask


class SequenceDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids, attention_mask = encode_text(self.texts[idx], self.vocab, self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BiLSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(pooled))


def compute_metrics(y_true, y_pred, logits=None):
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for idx, label_name in LABEL_MAPPING.items():
        per_class[label_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(support[idx]),
        }

    metrics = {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "negative_f1": float(f1_per_class[1]),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING)],
            output_dict=True,
            zero_division=0,
        ),
    }
    if logits is not None:
        metrics["logits_shape"] = list(logits.shape)
    return metrics


def evaluate_torch_model(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_logits),
    )


def run_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validate"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    val_loss = total_loss / len(data_loader)
    val_f1 = f1_score(all_labels, all_predictions, average="weighted")
    return val_loss, val_f1


def train_bilstm(data_splits, args, output_dir, checkpoint_dir, device):
    train_texts = data_splits["train"]["text"].values
    train_labels = data_splits["train"]["label"].values
    val_texts = data_splits["val"]["text"].values
    val_labels = data_splits["val"]["label"].values
    test_texts = data_splits["test"]["text"].values
    test_labels = data_splits["test"]["label"].values

    vocab = build_vocab(train_texts, min_freq=args.min_freq, max_vocab_size=args.max_vocab_size)
    print(f"Vocab size: {len(vocab)}")

    train_loader = DataLoader(
        SequenceDataset(train_texts, train_labels, vocab, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SequenceDataset(val_texts, val_labels, vocab, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SequenceDataset(test_texts, test_labels, vocab, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        dropout=args.dropout,
    ).to(device)

    class_weights = calculate_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = -float("inf")
    best_model_path = None

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = os.path.join(checkpoint_dir, f"bilstm_epoch{epoch + 1}_f1{val_f1:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "history": history,
                    "vocab": vocab,
                    "model_config": {
                        "embedding_dim": args.embedding_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "max_length": args.max_length,
                    },
                },
                best_model_path,
            )
            print(f"Best model saved: {best_model_path}")

    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    plot_training_history(history, save_path=os.path.join(output_dir, "training_history.png"))
    y_true, y_pred, logits = evaluate_torch_model(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, logits)

    predictions_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": [LABEL_MAPPING[i] for i in y_true],
            "predicted_label": [LABEL_MAPPING[i] for i in y_pred],
            "true_id": y_true,
            "predicted_id": y_pred,
            "logit_positive": logits[:, 0],
            "logit_negative": logits[:, 1],
            "logit_notr": logits[:, 2],
        }
    )

    extra_config = {
        "model_type": "bilstm",
        "vocab_size": len(vocab),
        "min_freq": args.min_freq,
        "max_vocab_size": args.max_vocab_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "best_model_path": best_model_path,
        "history": history,
    }
    return metrics, predictions_df, extra_config


def ensure_fasttext_vectors(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    vector_path = os.path.join(cache_dir, "cc.tr.300.vec.gz")
    temp_path = vector_path + ".part"

    if is_valid_gzip(vector_path):
        print(f"fastText vectors found: {vector_path}")
        return vector_path

    print(f"Downloading fastText Turkish vectors: {FASTTEXT_URL}")
    print(f"Destination: {vector_path}")
    download_with_resume(FASTTEXT_URL, temp_path)
    if not is_valid_gzip(temp_path):
        raise ContentTooShortError("Downloaded fastText file is not a valid gzip archive", None)
    os.replace(temp_path, vector_path)
    return vector_path


def is_valid_gzip(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False


def download_with_resume(url, destination, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        existing_size = os.path.getsize(destination) if os.path.exists(destination) else 0
        headers = {}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        print(f"Download attempt {attempt}/{max_attempts}; existing_bytes={existing_size}")
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=60) as response, open(destination, mode) as f:
                status = getattr(response, "status", None)
                if existing_size > 0 and status == 200:
                    print("Server ignored Range header; restarting download")
                    f.close()
                    os.remove(destination)
                    return download_with_resume(url, destination, max_attempts=max_attempts)

                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as exc:
            print(f"Download attempt failed: {exc}")
            if attempt == max_attempts:
                raise
            continue

        if is_valid_gzip(destination):
            print(f"Download complete: {destination}")
            return

    raise ContentTooShortError("Unable to complete valid fastText gzip download", None)


def build_fasttext_vocabulary(*text_arrays):
    vocab = set()
    for texts in text_arrays:
        for text in texts:
            for token in tokenize_text(text):
                vocab.add(token)
                vocab.add(token.lower())
    return vocab


def load_needed_fasttext_vectors(vector_path, needed_vocab):
    vectors = {}
    expected_dim = None

    with gzip.open(vector_path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
        if len(header) == 2 and header[0].isdigit():
            expected_dim = int(header[1])

        for line in tqdm(f, desc="Load fastText vectors"):
            parts = line.rstrip().split(" ")
            if len(parts) < 301:
                continue
            token = parts[0]
            if token not in needed_vocab:
                continue
            values = np.asarray(parts[1:], dtype=np.float32)
            if expected_dim is None:
                expected_dim = len(values)
            vectors[token] = values

    if expected_dim is None:
        expected_dim = 300

    print(f"Loaded fastText vectors: {len(vectors)}")
    return vectors, expected_dim


def sentence_vector(text, vectors, vector_dim):
    rows = []
    for token in tokenize_text(text):
        vector = vectors.get(token)
        if vector is None:
            vector = vectors.get(token.lower())
        if vector is not None:
            rows.append(vector)
    if not rows:
        return np.zeros(vector_dim, dtype=np.float32)
    return np.mean(rows, axis=0).astype(np.float32)


def vectorize_texts(texts, vectors, vector_dim):
    matrix = np.zeros((len(texts), vector_dim), dtype=np.float32)
    for idx, text in enumerate(tqdm(texts, desc="Vectorize texts")):
        matrix[idx] = sentence_vector(text, vectors, vector_dim)
    return matrix


def run_fasttext_logreg(data_splits, args, output_dir):
    train_texts = data_splits["train"]["text"].values
    train_labels = data_splits["train"]["label"].values
    val_texts = data_splits["val"]["text"].values
    val_labels = data_splits["val"]["label"].values
    test_texts = data_splits["test"]["text"].values
    test_labels = data_splits["test"]["label"].values

    vector_path = args.fasttext_path or ensure_fasttext_vectors(args.fasttext_cache_dir)
    needed_vocab = build_fasttext_vocabulary(train_texts, val_texts, test_texts)
    vectors, vector_dim = load_needed_fasttext_vectors(vector_path, needed_vocab)

    x_train = vectorize_texts(train_texts, vectors, vector_dim)
    x_val = vectorize_texts(val_texts, vectors, vector_dim)
    x_test = vectorize_texts(test_texts, vectors, vector_dim)

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=args.logreg_max_iter,
        random_state=args.seed,
    )
    classifier.fit(x_train, train_labels)

    train_pred = classifier.predict(x_train)
    val_pred = classifier.predict(x_val)
    test_pred = classifier.predict(x_test)

    metrics = compute_metrics(test_labels, test_pred)
    metrics["train_weighted_f1"] = float(f1_score(train_labels, train_pred, average="weighted"))
    metrics["val_weighted_f1"] = float(f1_score(val_labels, val_pred, average="weighted"))

    predictions_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": [LABEL_MAPPING[i] for i in test_labels],
            "predicted_label": [LABEL_MAPPING[i] for i in test_pred],
            "true_id": test_labels,
            "predicted_id": test_pred,
        }
    )

    model_path = os.path.join(output_dir, "model_bundle.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": classifier,
                "vector_dim": vector_dim,
                "fasttext_source": vector_path,
            },
            f,
        )

    extra_config = {
        "model_type": "fasttext_logreg",
        "fasttext_url": FASTTEXT_URL,
        "fasttext_path": vector_path,
        "loaded_vectors": len(vectors),
        "needed_vocab_size": len(needed_vocab),
        "vector_dim": vector_dim,
        "classifier": "LogisticRegression(class_weight=balanced)",
        "model_path": model_path,
    }
    return metrics, predictions_df, extra_config


def write_notes(path, args, metrics, extra_config):
    lines = [
        "# Experiment Notes",
        "",
        f"- Status: {args.run_status}",
        f"- Model: {args.model}",
        "- Dataset: winvoker/turkish-sentiment-analysis-dataset",
        "- Split seed: 42",
        f"- Training seed: {args.seed}",
        "- Preprocessing: manuscript-aligned shared cleaning",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Negative F1: {metrics['negative_f1']:.4f}",
        "",
        "## Model Details",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in extra_config.items() if key != "history")
    lines.extend(
        [
            "",
            "## Manuscript Integration",
            "",
            "- Stage 2 result is single-seed by protocol.",
            "- Eligible for manuscript after artifact verification and user approval.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 non-transformer baselines")
    parser.add_argument("--model", choices=["bilstm", "fasttext_logreg"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_status", choices=["dev", "candidate", "paper"], default="dev")
    parser.add_argument("--exp_id", type=str, default=None)

    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=100000)

    parser.add_argument("--fasttext_cache_dir", type=str, default="outputs/cache/fasttext")
    parser.add_argument("--fasttext_path", type=str, default=None)
    parser.add_argument("--logreg_max_iter", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)

    default_exp_ids = {
        "bilstm": f"s2_neural_bilstm_winvoker_seed{args.seed}_base",
        "fasttext_logreg": f"s2_embed_fasttext_logreg_winvoker_seed{args.seed}_base",
    }
    exp_id = args.exp_id or default_exp_ids[args.model]
    output_dir = os.path.join(ROOT_DIR, "outputs", "experiments", exp_id)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = load_winvoker_dataset()
    data_splits = preprocess_dataset(dataset, test_size=0.2, val_size=0.1, random_state=42)

    if args.model == "bilstm":
        metrics, predictions_df, extra_config = train_bilstm(
            data_splits,
            args,
            output_dir,
            checkpoint_dir,
            device,
        )
    else:
        metrics, predictions_df, extra_config = run_fasttext_logreg(data_splits, args, output_dir)

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        label_mapping=LABEL_MAPPING,
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )

    config = {
        "exp_id": exp_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "seed": args.seed,
        "dataset": "winvoker/turkish-sentiment-analysis-dataset",
        "split_seed": 42,
        "preprocessing": "manuscript_aligned_cleaning",
        "run_status": args.run_status,
        "single_seed_stage2": True,
        "device": str(device),
        "params": vars(args),
        "model_details": extra_config,
    }

    write_json(os.path.join(output_dir, "config.json"), config)
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    write_notes(os.path.join(output_dir, "notes.md"), args, metrics, extra_config)

    print(json.dumps({"exp_id": exp_id, "output_dir": output_dir, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
