"""
Docker-first baseline experiment runner.

Runs classical baseline models under locked experiment protocol and saves
paper-ready artifacts under outputs/experiments/<exp_id>/.
"""
import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import load_winvoker_dataset, preprocess_dataset


LABEL_MAPPING = {0: "Positive", 1: "Negative", 2: "Notr"}


def build_model(model_name, seed):
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    if model_name == "svm":
        return LinearSVC(class_weight="balanced", random_state=seed)
    if model_name == "nb":
        return MultinomialNB()
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_vectorizer(max_features):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
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

    return {
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
        ),
    }


def plot_confusion_matrix(cm, save_path):
    cm = np.array(cm)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING)],
        yticklabels=[LABEL_MAPPING[i] for i in sorted(LABEL_MAPPING)],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_notes(path, args, metrics):
    lines = [
        "# Experiment Notes",
        "",
        f"- Status: {args.run_status}",
        f"- Model: {args.model}",
        f"- Dataset: winvoker/turkish-sentiment-analysis-dataset",
        f"- Split seed: 42",
        f"- Training seed: {args.seed}",
        f"- Preprocessing: manuscript-aligned shared cleaning",
        f"- Vectorizer: TF-IDF (1,2)-gram, max_features={args.max_features}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Negative F1: {metrics['negative_f1']:.4f}",
        "",
        "## Manuscript Integration",
        "",
        "- Eligible for manuscript only after protocol-consistent rerun / seed policy satisfied.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiment and save artifacts")
    parser.add_argument("--model", choices=["logreg", "svm", "nb", "rf"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=10000)
    parser.add_argument("--run_status", choices=["dev", "candidate", "paper"], default="dev")
    parser.add_argument("--exp_id", type=str, default=None)
    args = parser.parse_args()

    exp_id = args.exp_id or f"s1_tfidf_{args.model}_winvoker_seed{args.seed}_base"
    output_dir = os.path.join("outputs", "experiments", exp_id)
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_winvoker_dataset()
    data_splits = preprocess_dataset(dataset, test_size=0.2, val_size=0.1, random_state=42)

    vectorizer = build_vectorizer(args.max_features)
    model = build_model(args.model, args.seed)

    train_texts = data_splits["train"]["text"].values
    train_labels = data_splits["train"]["label"].values
    val_texts = data_splits["val"]["text"].values
    val_labels = data_splits["val"]["label"].values
    test_texts = data_splits["test"]["text"].values
    test_labels = data_splits["test"]["label"].values

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    model.fit(x_train, train_labels)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    metrics = compute_metrics(test_labels, test_pred)
    metrics["train_weighted_f1"] = float(f1_score(train_labels, train_pred, average="weighted"))
    metrics["val_weighted_f1"] = float(f1_score(val_labels, val_pred, average="weighted"))

    config = {
        "exp_id": exp_id,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "seed": args.seed,
        "dataset": "winvoker/turkish-sentiment-analysis-dataset",
        "split_seed": 42,
        "preprocessing": "manuscript_aligned_cleaning",
        "vectorizer": {
            "type": "tfidf",
            "max_features": args.max_features,
            "ngram_range": [1, 2],
            "min_df": 2,
        },
        "run_status": args.run_status,
    }

    predictions_df = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": [LABEL_MAPPING[i] for i in test_labels],
            "predicted_label": [LABEL_MAPPING[i] for i in test_pred],
            "true_id": test_labels,
            "predicted_id": test_pred,
        }
    )

    with open(os.path.join(output_dir, "model_bundle.pkl"), "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": model}, f)

    write_json(os.path.join(output_dir, "config.json"), config)
    write_json(os.path.join(output_dir, "metrics.json"), metrics)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], os.path.join(output_dir, "confusion_matrix.png"))
    write_notes(os.path.join(output_dir, "notes.md"), args, metrics)

    print(json.dumps({"exp_id": exp_id, "output_dir": output_dir, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
