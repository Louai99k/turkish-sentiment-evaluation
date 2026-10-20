"""Validate frozen samples, then create aligned BERTurk/TF-IDF t-SNE figures.

Validate loads no dataset or model. Project fits only unsupervised text transforms
on the locked training split and uses the same frozen 3,000 test examples for
both representations. Figures are descriptive, not comparative performance tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_constants import WINVOKER_ASSIGNMENTS_PATH, WINVOKER_ASSIGNMENTS_SHA256, WINVOKER_DATASET_ID, WINVOKER_DATASET_REVISION

CONFIG = ROOT / "configs/analysis/berturk_representation_projection_v1.json"
GATE = ROOT / "outputs/audits/berturk_representation_projection_static_v2"
OUTPUT = ROOT / "outputs/analysis/berturk_representation_projection_v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_manifest(directory, expected_sha256=None):
    path = directory / "artifact_manifest.json"
    if expected_sha256 is not None:
        require(sha256_file(path) == expected_sha256, f"Frozen manifest changed: {directory}")
    manifest = read(path)
    require(manifest["run_status"] == "complete" and read(directory / "run_status.json")["status"] == "complete", f"Incomplete input: {directory}")
    for entry in manifest["files"]:
        artifact = (directory / entry["relative_to_run"]).resolve()
        require(artifact.is_relative_to(directory.resolve()), "Manifest path escapes input directory")
        require(artifact.stat().st_size == entry["size_bytes"] and sha256_file(artifact) == entry["sha256"], f"Input hash/size mismatch: {artifact}")


def inputs(config):
    extraction = ROOT / config["extraction_directory"]
    verified_manifest(extraction, config["extraction_manifest_sha256"])
    summary = read(extraction / "extraction_summary.json")
    require(summary["training"] is False and summary["all_labels_reproduced"] is True, "Extraction did not reproduce labels")
    require(summary["representation_shape"] == [3000, 768] and summary["samples_reproduced"] == 3060, "Unexpected extraction scope")
    require(summary["pooling"] == "pooler_output" and summary["attention_backend"] == "eager", "Unexpected extraction representation")
    require(summary["maximum_absolute_logit_difference"] <= 1e-4, "Saved-logit reproduction failed")
    index = pd.read_csv(extraction / "sample_index.csv", keep_default_na=False).iloc[:3000].copy()
    sample_path = ROOT / config["representation_sample"]
    require(sha256_file(sample_path) == config["representation_sample_sha256"], "Frozen representation sample changed")
    sample = pd.read_csv(sample_path, keep_default_na=False)
    require(len(index) == len(sample) == 3000, "Incorrect representation sample length")
    require(index.sample_id.is_unique and sample.sample_id.is_unique, "Duplicate sample IDs")
    for field in ("sample_id", "test_position", "source", "true_id"):
        require(index[field].tolist() == sample[field].tolist(), f"Representation alignment mismatch: {field}")
    require(sample.true_id.value_counts().to_dict() == {0: 1000, 1: 1000, 2: 1000}, "Class balance changed")
    require([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in sample.text] == sample.cleaned_text_sha256.tolist(), "Frozen sample text hashes changed")
    vectors = np.load(extraction / "berturk_pooler_output.npy", mmap_mode="r", allow_pickle=False)
    require(vectors.shape == (3000, 768) and vectors.dtype == np.float32 and np.isfinite(vectors).all(), "Invalid BERTurk matrix")
    profile_path = ROOT / config["profile_path"]
    fingerprint_path = ROOT / config["fingerprints_path"]
    require(sha256_file(profile_path) == config["profile_sha256"], "Frozen preprocessing profile changed")
    require(sha256_file(fingerprint_path) == config["fingerprints_sha256"], "Frozen profile fingerprints changed")
    fingerprint = read(fingerprint_path)
    require(fingerprint["dataset_id"] == WINVOKER_DATASET_ID and fingerprint["dataset_revision"] == WINVOKER_DATASET_REVISION, "Dataset identity mismatch")
    require(fingerprint["split_assignments_sha256"] == WINVOKER_ASSIGNMENTS_SHA256, "Fingerprint split identity mismatch")
    require(fingerprint["profiles"][config["profile_id"]]["profile_config_sha256"] == config["profile_sha256"], "Fingerprint profile identity mismatch")
    train = fingerprint["profiles"][config["profile_id"]]["splits"]["train"]
    require(train == {"rows": config["train_rows"], "ordered_cleaned_text_sha256": config["train_ordered_cleaned_text_sha256"]}, "Training fingerprint contract mismatch")
    extraction_config = read(extraction / "config.json")["extraction_config"]["tfidf_followup"]
    for key, expected in config["tfidf"].items():
        require(extraction_config[key] == expected, f"TF-IDF specification changed: {key}")
    for key, extraction_key in (("n_components", "svd_components"), ("algorithm", "svd_algorithm"),
                                ("n_iter", "svd_n_iter"), ("random_state", "random_state")):
        require(config["svd"][key] == extraction_config[extraction_key], f"SVD specification changed: {key}")
    plan = read(ROOT / "outputs/audits/berturk_qualitative_plan_v1/frozen_plan.json")
    require(config["tsne"] == plan["representation_visualization"]["tsne"], "Frozen t-SNE settings changed")
    require(config["figure"]["class_names"] == ["Positive", "Negative", "Notr"], "Class display mapping changed")
    require(sha256_file(WINVOKER_ASSIGNMENTS_PATH) == WINVOKER_ASSIGNMENTS_SHA256, "Locked split assignments changed")
    identity = {"config_sha256": sha256_file(CONFIG), "runner_sha256": sha256_file(Path(__file__)),
                "artifact_writer_sha256": sha256_file(ROOT / "src/experiment_artifacts.py"),
                "data_loader_sha256": sha256_file(ROOT / "src/data_loader.py"),
                "preprocessing_source_sha256": sha256_file(ROOT / "src/preprocessing.py"),
                "extraction_manifest_sha256": config["extraction_manifest_sha256"],
                "sample_sha256": config["representation_sample_sha256"],
                "profile_sha256": config["profile_sha256"], "fingerprints_sha256": config["fingerprints_sha256"],
                "assignments_sha256": WINVOKER_ASSIGNMENTS_SHA256}
    return sample, vectors, identity


def validate(run, config):
    sample, vectors, identity = inputs(config)
    run.write_json("gate.json", {"status": "passed", "identity": identity,
                                "sample_count": len(sample), "berturk_shape": list(vectors.shape),
                                "dataset_loaded": False, "tfidf_fitted": False})
    run.write_text("report.md", "# Representation projection preflight\n\nFrozen extraction and sample hashes, 3,000 ordered sample IDs, class balance, text hashes, BERTurk matrix and training fingerprint contract passed. No dataset or text transform was loaded/fitted.\n")
    run.declare_complete(["gate.json", "report.md"])


def draw_figure(run, points, sample, config, color_by, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    values = sample[color_by].tolist()
    names = config["figure"]["class_names"] if color_by == "true_id" else sorted(set(values))
    palette = plt.get_cmap("tab10")
    for axis, representation in zip(axes, ("BERTurk pooler output", "TF-IDF + training-fit SVD")):
        xy = points[representation]
        for index, label in enumerate(range(3) if color_by == "true_id" else names):
            mask = np.asarray([value == label for value in values], dtype=bool)
            axis.scatter(xy[mask, 0], xy[mask, 1], s=config["figure"]["point_size"],
                         alpha=config["figure"]["point_alpha"], color=palette(index),
                         label=names[index] if color_by == "true_id" else label, rasterized=True)
        axis.set_title(representation)
        axis.set_xlabel("t-SNE dimension 1")
        axis.set_ylabel("t-SNE dimension 2")
        axis.legend(markerscale=2, fontsize=8)
    fig.suptitle("Same 3,000 selected test examples; independent t-SNE coordinates")
    fig.savefig(run.path(filename), dpi=config["figure"]["dpi"])
    plt.close(fig)


def project(run, config):
    sample, vectors, identity = inputs(config)
    verified_manifest(GATE)
    gate = read(GATE / "gate.json")
    require(gate["status"] == "passed" and gate["identity"] == identity, "Static preflight missing or stale")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from sklearn import __version__ as sklearn_version
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    from src.data_loader import load_winvoker_dataset, preprocess_dataset_with_profile
    from src.preprocessing import load_preprocessing_profile, ordered_cleaned_text_sha256

    _, profile = load_preprocessing_profile(config["profile_path"])
    require(profile["profile_id"] == config["profile_id"], "Wrong preprocessing profile")
    dataset = load_winvoker_dataset(dataset_name=WINVOKER_DATASET_ID, revision=WINVOKER_DATASET_REVISION)
    train = preprocess_dataset_with_profile(dataset, profile, included_splits=("train",))["train"]
    del dataset
    require(len(train) == config["train_rows"], "Training-row count changed")
    actual_fingerprint = ordered_cleaned_text_sha256(train.text)
    require(actual_fingerprint == config["train_ordered_cleaned_text_sha256"], "Training text fingerprint mismatch")
    print(f"Locked training split verified: {len(train)} texts", flush=True)

    tf = config["tfidf"]
    vectorizer = TfidfVectorizer(analyzer=tf["analyzer"], ngram_range=tuple(tf["ngram_range"]),
        lowercase=tf["lowercase"], token_pattern=tf["token_pattern"], strip_accents=tf["strip_accents"],
        stop_words=tf["stop_words"], max_features=tf["max_features"], min_df=tf["min_df"],
        max_df=tf["max_df"], norm=tf["norm"], use_idf=tf["use_idf"], smooth_idf=tf["smooth_idf"],
        sublinear_tf=tf["sublinear_tf"], dtype=np.dtype(tf["dtype"]).type)
    x_train = vectorizer.fit_transform(train.text)
    del train
    require(x_train.shape[0] == config["train_rows"] and x_train.shape[1] >= config["svd"]["n_components"], "Invalid TF-IDF training matrix")
    print(f"Training-only TF-IDF fitted: {x_train.shape[1]} terms", flush=True)
    sv = config["svd"]
    reducer = TruncatedSVD(n_components=sv["n_components"], algorithm=sv["algorithm"],
                           n_iter=sv["n_iter"], random_state=sv["random_state"])
    reducer.fit(x_train)
    del x_train
    x_test = vectorizer.transform(sample.text)
    tfidf_vectors = reducer.transform(x_test).astype(np.float32)
    require(tfidf_vectors.shape == (3000, 100) and np.isfinite(tfidf_vectors).all(), "Invalid selected TF-IDF/SVD matrix")
    np.save(run.path("tfidf_trainfit_svd_sample.npy"), tfidf_vectors, allow_pickle=False)
    print("Training-only SVD fitted and frozen sample transformed", flush=True)

    tsne_params = config["tsne"]
    points = {}
    for label, matrix in (("BERTurk pooler output", np.asarray(vectors)),
                          ("TF-IDF + training-fit SVD", tfidf_vectors)):
        points[label] = TSNE(**tsne_params).fit_transform(matrix)
        require(points[label].shape == (3000, 2) and np.isfinite(points[label]).all(), f"Invalid t-SNE coordinates: {label}")
        print(f"t-SNE completed: {label}", flush=True)
    rows = []
    for label, xy in points.items():
        for i, row in sample.reset_index(drop=True).iterrows():
            rows.append({"representation": label, "sample_id": row.sample_id,
                         "test_position": int(row.test_position), "source": row.source,
                         "true_id": int(row.true_id), "tsne_x": float(xy[i, 0]), "tsne_y": float(xy[i, 1])})
    run.write_csv("projection_points.csv", rows)
    draw_figure(run, points, sample, config, "true_id", "tsne_by_class.png")
    draw_figure(run, points, sample, config, "source", "tsne_by_source.png")
    run.write_json("projection_summary.json", {"identity": identity, "training_rows": config["train_rows"],
        "training_ordered_cleaned_text_sha256": actual_fingerprint, "selected_sample_rows": len(sample),
        "tfidf_features": int(x_test.shape[1]), "selected_tfidf_nonzero": int(x_test.nnz),
        "svd_explained_variance_ratio_sum": float(reducer.explained_variance_ratio_.sum()),
        "svd_components_shape": list(reducer.components_.shape),
        "tfidf_vocabulary_sha256": hashlib.sha256(json.dumps(
            {term: int(index) for term, index in vectorizer.vocabulary_.items()},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "sklearn_version": sklearn_version, "numpy_version": np.__version__,
        "interpretation": config["interpretation"]})
    run.write_text("report.md", "# Aligned representation projections\n\nThe same 3,000 frozen, class-balanced test examples appear in both panels. TF-IDF and SVD were fitted on the 317,288-row locked training split only. Both t-SNE projections were fitted independently to the selected examples, so axes and distances are not directly comparable between panels. The views are descriptive, and source-label composition may drive apparent clusters. This is a new TF-IDF representation for visualization, not the historical SVM feature matrix.\n")
    run.declare_complete(["tfidf_trainfit_svd_sample.npy", "projection_points.csv", "tsne_by_class.png", "tsne_by_source.png", "projection_summary.json", "report.md"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "project"), required=True)
    args = parser.parse_args()
    config = read(CONFIG)
    require(config["schema_version"] == "berturk-representation-projection-v1", "Unknown projection contract")
    output = GATE if args.stage == "validate" else OUTPUT
    require(not output.exists(), f"Output already exists; preserve evidence: {output}")
    with ExperimentRun(output, run_id=output.name, run_kind=f"berturk_representation_{args.stage}",
                       config={"stage": args.stage, "projection_config": config, "config_sha256": sha256_file(CONFIG)}) as run:
        if args.stage == "validate":
            validate(run, config)
        else:
            project(run, config)


if __name__ == "__main__":
    main()
