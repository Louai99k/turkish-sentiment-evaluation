"""Prepare or explicitly fit ONE independently identified reported configuration.

Default is preparation only (hash checks, no data/model loading). Actual fitting
requires --mode fit AND --confirm-new-fit plus unchanged prior preparation.
No sweep, download, historical gate bypass, original-ID reuse or automatic resume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reproduction import local, read, resolve_setup, verify_inputs


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, type):
        return {"python_type": value.__module__ + "." + value.__qualname__}
    if hasattr(value, "item"):
        return value.item()
    return value


def classical_model(config):
    """Instantiate only when fitting is explicitly requested; never during prepare."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    constructors = {"tfidf_logreg": LogisticRegression, "tfidf_svm": LinearSVC,
                    "tfidf_nb": MultinomialNB, "tfidf_rf": RandomForestClassifier,
                    "fasttext_logreg": LogisticRegression}
    parameters = config["parameters"]
    arguments = parameters.get("constructor_arguments", parameters.get("classifier_arguments"))
    return constructors[config["setup"]](**arguments)


def fit_classical(config, train, run):
    import pickle
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    parameters = config["parameters"]
    estimator = classical_model(config)
    if config["family"] == "classical":
        arguments = dict(parameters["tfidf_constructor"])
        arguments["ngram_range"] = tuple(arguments["ngram_range"])
        vectorizer = TfidfVectorizer(**arguments)
        features = vectorizer.fit_transform(train["text"])
        transform = vectorizer.transform
        bundle = {"vectorizer": vectorizer}
        effective_features = plain(vectorizer.get_params())
    else:
        from scripts.run_fasttext_train_vocab_refit import freeze_train_vocabulary, load_vectors, vectorize
        vocabulary = freeze_train_vocabulary(train["text"])
        vectors, record = load_vectors(Path(config["fasttext_archive"]), vocabulary, 300, 2000000)
        features, train_coverage = vectorize(train["text"], vectors, 300)

        def transform(texts):
            return vectorize(texts, vectors, 300)[0]

        bundle = {"vectors": vectors, "vector_dim": 300, "training_vocabulary": sorted(vocabulary)}
        effective_features = {"vector_subset": record, "train_coverage": train_coverage}
    estimator.fit(features, train["label"].to_numpy())
    if estimator.classes_.tolist() != [0, 1, 2]:
        raise ValueError("Unexpected fitted class ordering")
    bundle.update(model=estimator, configuration=config)
    with run.path("model_bundle.pkl").open("xb") as stream:
        pickle.dump(bundle, stream)
    run.write_json("effective_parameters.json", {"estimator": plain(estimator.get_params()),
                                               "features": effective_features})

    def predict(frame):
        matrix = transform(frame["text"])
        if hasattr(estimator, "decision_function"):
            scores, kind = estimator.decision_function(matrix), "decision_score"
        elif hasattr(estimator, "predict_log_proba"):
            scores, kind = estimator.predict_log_proba(matrix), "log_probability"
        else:
            scores, kind = estimator.predict_proba(matrix), "probability"
        return frame["label"].to_numpy(dtype=np.int64), estimator.predict(matrix), scores, kind

    return predict, {"path": "model_bundle.pkl", "selection": "one fixed fit, no checkpoint selection"}


def fit_neural(config, train_val, run, device):
    import torch
    from torch.utils.data import DataLoader
    from scripts.run_berturk_experiment import collect_predictions
    from src.data_loader import calculate_class_weights, create_data_loaders, TurkishSentimentDataset
    parameters = config["parameters"]
    lineage = {"source_row_key": "source_row_index",
               "sample_id_function": "src.protocol_predictions.stable_winvoker_sample_id"}
    metadata = {"replication_configuration": config, "stable_sample_lineage": lineage,
                "selection_metric": "validation_weighted_f1", "tie_policy": "retain_earliest_epoch"}
    if config["family"] == "transformer":
        from transformers import AutoTokenizer
        from src.transformer_model import get_transformer_model
        from src.train import Trainer
        model_config = config["model_config"]
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_id"],
                        revision=model_config["tokenizer_revision"], local_files_only=True)
        loaders = create_data_loaders(train_val, tokenizer, batch_size=parameters["batch_size"],
                                      max_length=parameters["max_length"], seed=config["training_seed"])
        model = get_transformer_model(model_config, dropout=parameters["dropout"],
                                      device=device, local_files_only=True)
        metadata.update(pooling=model_config["project_head"]["pooling"], backbone_access="model.backbone",
                        tokenizer_id=model_config["tokenizer_id"], tokenizer_revision=model_config["tokenizer_revision"])
        trainer = Trainer(model, loaders["loaders"]["train"], loaders["loaders"]["val"],
                          loaders["class_weights"], device, learning_rate=parameters["learning_rate"],
                          epochs=parameters["epochs"], loss_mode=parameters["loss"],
                          output_dir=run.path("checkpoints"), history_path=run.path("history.json"),
                          checkpoint_metadata=metadata)
        history = trainer.train(early_stopping_patience=parameters["early_stopping_patience"])
        best = trainer.load_best_model()  # loads only this run's freshly created local checkpoint
        selected = {"path": "checkpoints/best_checkpoint.pt", "selected_epoch_zero_based": int(best["epoch"]),
                    "validation_weighted_f1": float(best["val_f1"]), **metadata}

        def loader(frame):
            return DataLoader(TurkishSentimentDataset(frame["text"].to_numpy(), frame["label"].to_numpy(),
                              tokenizer, parameters["max_length"]), batch_size=parameters["batch_size"],
                              shuffle=False, num_workers=0)
    else:
        from scripts.run_bilstm_protocol_v2 import save_checkpoint
        from scripts.run_stage2_neural_baselines import (
            BiLSTMSentimentClassifier, SequenceDataset, build_vocab, run_epoch, validate_epoch,
        )
        vocabulary = build_vocab(train_val["train"]["text"].to_numpy(),
                                 min_freq=parameters["min_freq"], max_vocab_size=parameters["max_vocab_size"])
        generator = torch.Generator().manual_seed(config["training_seed"])

        def loader(frame, shuffle=False):
            return DataLoader(SequenceDataset(frame["text"].to_numpy(), frame["label"].to_numpy(),
                              vocabulary, parameters["max_length"]), batch_size=parameters["batch_size"],
                              shuffle=shuffle, generator=generator if shuffle else None, num_workers=0)

        train_loader, val_loader = loader(train_val["train"], True), loader(train_val["val"])
        model = BiLSTMSentimentClassifier(len(vocabulary), parameters["embedding_dim"],
                  parameters["hidden_dim"], parameters["num_classes"], parameters["dropout"]).to(device)
        weights = calculate_class_weights(train_val["train"]["label"].to_numpy()).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"],
                                     weight_decay=parameters["weight_decay"], eps=parameters["adam_epsilon"])
        history, best_f1 = {"train_loss": [], "val_loss": [], "val_f1": []}, -float("inf")
        metadata.update(pooling="concatenated_final_bidirectional_lstm_hidden_states",
                        backbone_access="model.embedding and model.lstm", tokenizer="str(text).split()",
                        vocabulary=vocabulary)
        checkpoint = run.path("checkpoints/best_checkpoint.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(parameters["epochs"]):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device)
            for key, value in (("train_loss", train_loss), ("val_loss", val_loss), ("val_f1", val_f1)):
                history[key].append(float(value))
            if val_f1 > best_f1:  # exact ties retain the earlier checkpoint
                best_f1 = float(val_f1)
                save_checkpoint(checkpoint, {"schema_version": "independent-bilstm-checkpoint-v1", "epoch": epoch,
                            "val_f1": best_f1, "model_state_dict": model.state_dict(), "metadata": metadata})
            run.write_json("history.json", history)
            # Retain recovery state; automatic restart/resume is intentionally unsupported.
            save_checkpoint(run.path("checkpoints/last_epoch_checkpoint.pt"),
                       {"completed_epochs": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "history": history,
                        "torch_rng_state": torch.get_rng_state(), "generator_state": generator.get_state(),
                        "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                        "metadata": metadata})
        best = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        selected = {"path": "checkpoints/best_checkpoint.pt", "selected_epoch_zero_based": int(best["epoch"]),
                    "validation_weighted_f1": best_f1, **metadata}
    run.write_json("history.json", history)

    def predict(frame):
        true, predicted, scores = collect_predictions(model, loader(frame), device)
        return true, predicted, scores, "logit"

    return predict, selected


def execute_fit(config):
    """ONLY called behind explicit fit flag, confirmation and unchanged preparation."""
    import numpy as np
    import pandas as pd
    from scripts.run_berturk_experiment import configure_reproducibility, resolve_device, runtime_environment
    from src.data_loader import load_winvoker_dataset, preprocess_dataset_with_profile
    from src.experiment_artifacts import ExperimentRun, sha256_file
    from src.preprocessing import (fingerprint_preprocessed_splits, load_preprocessing_profile,
                                   load_profile_fingerprint_manifest, preprocessing_profile_record, verify_split_fingerprints)
    from src.protocol_metrics import classification_metrics, build_evaluation_metrics
    from src.protocol_predictions import (build_winvoker_prediction_frame, load_winvoker_test_lineage,
                                           stable_winvoker_sample_id, sha256_text)
    reproducibility = configure_reproducibility(config["implementation_random_state"])
    device = resolve_device(config["device"])
    runtime = runtime_environment(device, reproducibility)
    runtime["source_identity_authority"] = "resolved configuration source_hashes; not an inherited parent Git checkout"
    path, profile = load_preprocessing_profile(config["preprocessing_profile"])
    record = preprocessing_profile_record(path, profile)
    _, _, expected = load_profile_fingerprint_manifest(config["preprocessing_fingerprints"], record)
    with ExperimentRun(ROOT / "outputs/experiments" / config["run_id"], run_id=config["run_id"],
                       run_kind="independent_fixed_setup_replication", config=config,
                       runtime_environment=runtime) as run:
        dataset_config = read(local(config["dataset_config"]))
        dataset = load_winvoker_dataset(dataset_config["dataset_id"], dataset_config["dataset_revision"])
        train_val = preprocess_dataset_with_profile(dataset, profile, included_splits=("train", "val"))
        verify_split_fingerprints(fingerprint_preprocessed_splits(train_val), expected["splits"])
        if config["family"] in {"classical", "fasttext"}:
            predict, checkpoint = fit_classical(config, train_val["train"], run)
        else:
            predict, checkpoint = fit_neural(config, train_val, run, device)
        true, predicted, scores, kind = predict(train_val["val"])
        validation = train_val["val"]
        if not np.array_equal(true, validation["label"].to_numpy()) or np.asarray(scores).shape != (len(true), 3) or not np.isfinite(scores).all():
            raise ValueError("Validation score/label alignment differs")
        rows = pd.DataFrame({"source_row_index": validation["source_row_index"].to_numpy(),
                             "sample_id": [stable_winvoker_sample_id(value) for value in validation["source_row_index"]],
                             "cleaned_text_sha256": [sha256_text(value) for value in validation["text"]],
                             "source": validation["dataset"].to_numpy(), "true_id": true, "predicted_id": predicted})
        for column, label in enumerate(("positive", "negative", "notr")):
            rows[f"{kind}_{label}"] = scores[:, column]
        run.write_csv("validation_predictions.csv", rows)
        run.write_json("validation_metrics.json", classification_metrics(true, predicted, subset_id="validation"))
        checkpoint["sha256"] = sha256_file(run.path(checkpoint["path"]))
        run.write_json("checkpoint_metadata.json", checkpoint)
        required = ["validation_predictions.csv", "validation_metrics.json", "checkpoint_metadata.json", checkpoint["path"]]
        # Only this branch constructs held-out text, lineage, loader or predictions.
        if config["stage"] == "replication":
            test = preprocess_dataset_with_profile(dataset, profile, included_splits=("test",))["test"]
            verify_split_fingerprints(fingerprint_preprocessed_splits({"test": test}), expected["splits"])
            true, predicted, scores, kind = predict(test)
            predictions = build_winvoker_prediction_frame(load_winvoker_test_lineage(), test["text"].to_numpy(),
                                                          true, predicted, scores=scores, score_kind=kind)
            run.write_csv("predictions.csv", predictions)
            run.write_json("metrics.json", build_evaluation_metrics(predictions))
            required += ["predictions.csv", "metrics.json"]
        run.write_text("notes.md", "New independently identified replication; not an original fit or historical approval.\n"
                       "External checkpoints/masks and accepted experiments remain unchanged.\n")
        run.declare_complete([*required, "notes.md"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stage", choices=("validation_only", "replication"), default="validation_only")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("prepare", "fit"), default="prepare")
    parser.add_argument("--confirm-new-fit", action="store_true")
    parser.add_argument("--fasttext-archive", type=Path)
    args = parser.parse_args(argv)
    if not (ROOT / "workspace_receipt.json").is_file():
        raise ValueError("Run only in a NEW isolated execution workspace")
    if args.confirm_new_fit != (args.mode == "fit"):
        parser.error("Fit requires --mode fit AND --confirm-new-fit; preparation must not use confirmation")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

    def no_network(event, arguments):
        if event == "socket.connect":
            raise RuntimeError("Replication runner is offline; acquire inputs separately before execution")

    sys.addaudithook(no_network)
    config = resolve_setup(args.setup, run_id=args.run_id, seed=args.seed, stage=args.stage,
                           device=args.device, archive=args.fasttext_archive)
    verify_inputs(config)
    prepared = ROOT / "outputs/preparations" / args.run_id
    if args.mode == "prepare":
        if (ROOT / "outputs/experiments" / args.run_id).exists():
            raise ValueError("Existing experiment ID; never overwrite or restart it")
        prepared.mkdir(parents=True, exist_ok=False)
        with (prepared / "resolved_config.json").open("x", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        print(f"Prepared {args.run_id}; inputs/source identities verified. NO fitting, inference or data/model loading.")
        return
    if read(prepared / "resolved_config.json") != config:
        raise ValueError("Source/config/arguments changed since preparation; do not fit")
    if (ROOT / "outputs/experiments" / args.run_id).exists():
        raise ValueError("Existing experiment preserved; no overwrite/restart/resume supported by this new runner")
    execute_fit(config)


if __name__ == "__main__":
    main()
