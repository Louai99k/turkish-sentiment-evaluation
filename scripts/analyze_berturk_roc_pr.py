"""Generate audited ROC/PR summaries and figures from frozen saved logits on CPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_final_transformer_seeds import verify_manifest
from src.experiment_artifacts import ExperimentRun, sha256_file
from src.protocol_predictions import load_winvoker_test_lineage, parse_boolean_series, stable_winvoker_sample_id

CONFIG_PATH = ROOT / "configs/analysis/berturk_roc_pr_v1.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Independent empirical ROC-AUC calculation with average ranks for ties."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    require(positives > 0 and negatives > 0, "ROC requires both classes")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main() -> None:
    contract = read_json(CONFIG_PATH)
    output_dir = ROOT / "outputs/analysis" / contract["run_id"]
    require(not output_dir.exists(), f"Output already exists: {output_dir}")
    directory = ROOT / "outputs/experiments" / contract["experiment_id"]
    config = {
        **contract,
        "analysis_config_sha256": sha256_file(CONFIG_PATH),
        "analysis_script_sha256": sha256_file(Path(__file__)),
    }
    with ExperimentRun(
        output_dir, run_id=contract["run_id"], run_kind="saved_prediction_roc_pr_analysis",
        config=config, runtime_environment={"execution_kind": "cpu_only_saved_logits_postprocessing"},
    ) as run:
        require(sha256_file(ROOT / contract["selection_plan_manifest"]) == contract["selection_plan_manifest_sha256"], "Selection-plan manifest changed")
        selection_manifest = read_json(ROOT / contract["selection_plan_manifest"])
        selected_entry = next(item for item in selection_manifest["files"] if item["relative_to_run"] == "selected_checkpoint.json")
        require(sha256_file(ROOT / contract["selected_checkpoint_record"]) == selected_entry["sha256"], "Selection record hash differs")
        selection = read_json(ROOT / contract["selected_checkpoint_record"])
        require(selection["experiment_id"] == contract["experiment_id"] and selection["training_seed"] == contract["training_seed"], "Previously selected model differs")
        source_entries = verify_manifest(directory, contract["experiment_id"])
        prediction_path = directory / "predictions.csv"
        require(sha256_file(prediction_path) == contract["predictions_sha256"] == selection["predictions_sha256"], "Frozen prediction hash differs")
        source_config = read_json(directory / "config.json")
        require(source_config["seed"] == 52 and source_config["loss_mode"] == "unweighted", "Source configuration differs")
        require(source_config["label_order"] == contract["label_order"], "Source logit label order differs")
        require(source_config["model_revision"] == "b6e1de16c983e0f2c70664591ea3f22810072608", "Pinned backbone differs")
        require(source_config["max_length"] == 128 and source_config["dropout"] == 0.3, "Source pipeline differs")
        lineage = load_winvoker_test_lineage()
        frame = pd.read_csv(prediction_path, keep_default_na=False)
        require(len(frame) == len(lineage) and frame["sample_id"].is_unique, "Prediction support/uniqueness differs")
        require(np.array_equal(frame["sample_id"].to_numpy(), lineage["source_row_index"].map(stable_winvoker_sample_id).to_numpy()), "Sample-ID lineage differs")
        for actual_key, expected_key in (
            ("test_position", "test_position"), ("source_row_index", "source_row_index"),
            ("source", "dataset"), ("cleaned_text_sha256", "cleaned_text_sha256"), ("true_id", "label_id"),
        ):
            require(np.array_equal(frame[actual_key].to_numpy(), lineage[expected_key].to_numpy()), f"Ordered lineage differs: {actual_key}")
        strict = lineage["strict_included"].to_numpy(dtype=bool)
        require(np.array_equal(parse_boolean_series(frame["strict_included"], "strict_included").to_numpy(), strict), "Strict mask differs")
        logits = frame[contract["logit_columns"]].to_numpy(dtype=np.float64)
        require(logits.shape == (len(frame), 3) and np.isfinite(logits).all(), "Invalid saved logits")
        truth = frame["true_id"].to_numpy(dtype=np.int64)
        predicted = frame["predicted_id"].to_numpy(dtype=np.int64)
        require(np.array_equal(logits.argmax(axis=1), predicted), "Saved predictions disagree with logit argmax")
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        require(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0), "Softmax failed")
        rows: list[dict] = []
        curve_rows: list[dict] = []
        required = ["metrics.csv", "metrics.json", "curve_points.csv", "input_audit.json", "report.md"]
        run.path("figures").mkdir(exist_ok=True)
        for subset, mask in (
            ("historic_test_complete", np.ones(len(frame), dtype=bool)),
            ("historic_test_strict", strict),
        ):
            targets = np.eye(3, dtype=np.int64)[truth[mask]]
            scores = probabilities[mask]
            n = len(targets)
            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
            fig_pr, ax_pr = plt.subplots(figsize=(7, 5))
            class_rows = []
            for index, label in enumerate(contract["label_order"] + ["micro"]):
                y, s = (targets.ravel(), scores.ravel()) if label == "micro" else (targets[:, index], scores[:, index])
                fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
                precision, recall, _ = precision_recall_curve(y, s)
                roc_area = float(auc(fpr, tpr))
                rank_area = rank_auc(y, s)
                require(abs(roc_area - rank_area) <= 1e-12, f"ROC/rank AUC disagreement: {subset}/{label}")
                row = {
                    "subset_id": subset, "label_or_average": label, "rows": n,
                    "positive_support": int(y.sum()), "binary_observations": len(y),
                    "prevalence": float(y.mean()), "roc_auc": roc_area,
                    "roc_auc_rank_check": rank_area,
                    "pr_auc_trapezoidal": float(auc(recall, precision)),
                    "average_precision": float(average_precision_score(y, s)),
                }
                rows.append(row)
                if label != "micro":
                    class_rows.append(row)
                color = "black" if label == "micro" else ["#0072B2", "#D55E00", "#009E73"][index]
                display = "Neutral (Notr)" if label == "Notr" else label.capitalize() if label == "micro" else label
                ax_roc.plot(fpr, tpr, color=color, linewidth=1.6, label=f"{display}: AUC={roc_area:.4f}")
                ax_pr.plot(recall, precision, color=color, linewidth=1.6, label=f"{display}: AP={row['average_precision']:.4f}")
                if label != "micro":
                    ax_pr.axhline(row["prevalence"], color=color, linestyle=":", linewidth=0.8, alpha=0.65)
                for point, (x, y_value) in enumerate(zip(fpr, tpr)):
                    curve_rows.append({"subset_id": subset, "curve": "roc", "label": label, "point": point, "x": float(x), "y": float(y_value)})
                for point, (x, y_value) in enumerate(zip(recall, precision)):
                    curve_rows.append({"subset_id": subset, "curve": "pr", "label": label, "point": point, "x": float(x), "y": float(y_value)})
            for average in ("macro", "weighted"):
                weights = np.ones(3) / 3 if average == "macro" else np.array([r["positive_support"] for r in class_rows], dtype=float) / n
                rows.append({
                    "subset_id": subset, "label_or_average": average, "rows": n,
                    "positive_support": None, "binary_observations": None, "prevalence": None,
                    **{metric: float(np.dot(weights, [r[metric] for r in class_rows])) for metric in ("roc_auc", "pr_auc_trapezoidal", "average_precision")},
                })
            view = "complete test" if subset.endswith("complete") else "strict test"
            title = f"BERTurk, unweighted CE, seed 52 — {view} (n={n:,})"
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
            ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title=title, xlim=(0, 1), ylim=(0, 1.01))
            ax_pr.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.01))
            for kind, fig, ax in (("roc", fig_roc, ax_roc), ("pr", fig_pr, ax_pr)):
                ax.legend(loc="lower right" if kind == "roc" else "lower left", fontsize=8)
                ax.grid(alpha=0.2)
                fig.tight_layout()
                for extension in ("png", "pdf"):
                    relative = f"figures/{kind}_{subset}.{extension}"
                    fig.savefig(run.path(relative), dpi=300, bbox_inches="tight")
                    required.append(relative)
                plt.close(fig)
        run.write_csv("metrics.csv", pd.DataFrame(rows))
        run.write_json("metrics.json", {"label_order": contract["label_order"], "rows": rows, "representative_checkpoint_only": True})
        run.write_csv("curve_points.csv", pd.DataFrame(curve_rows))
        run.write_json("input_audit.json", {
            "source_manifest_entries_verified": source_entries,
            "source_manifest_sha256": sha256_file(directory / "artifact_manifest.json"),
            "prediction_sha256": sha256_file(prediction_path),
            "ordered_lineage_matches": True, "logit_argmax_matches": True,
            "complete_rows": len(frame), "strict_rows": int(strict.sum()),
            "roc_auc_rank_checks_passed": True,
        })
        run.write_text("report.md", (
            "# BERTurk ROC/PR analysis\n\n"
            "Source: accepted unweighted BERTurk seed 52, selected previously using validation. "
            "Complete and strict test views use saved logits transformed by stable float64 softmax. "
            "All source manifest hashes, ordered lineage, logit argmax, and ROC/rank-AUC checks passed.\n\n"
            "ROC uses one-vs-rest discrimination; macro is the arithmetic mean of class AUCs, "
            "weighted uses class supports, and micro flattens one-hot indicators and scores. "
            "PR trapezoidal area is distinct from noninterpolated average precision (AP); "
            "the PR plot legends show AP and dotted lines show class prevalence. "
            "These scores are not calibrated probabilities, and high Neutral AUC remains "
            "subject to the documented source/label confound.\n\n"
            "This is one representative checkpoint, not a five-seed mean or superiority test. "
            "No training, model/dataset loading, or inference occurred. Paper figures were not overwritten.\n"
        ))
        run.declare_complete(required)
        print(f"ROC/PR analysis completed: {contract['run_id']}; {len(frame):,} complete and {int(strict.sum()):,} strict rows")


if __name__ == "__main__":
    main()
