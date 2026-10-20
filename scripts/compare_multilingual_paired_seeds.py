"""Descriptive matched-seed mBERT/XLM-R comparison from accepted artifacts.

CPU-only post-processing. This is exploratory: the protocol's confirmatory
family compares weighted BERTurk with validation-selected competitors, not
the two multilingual candidates directly. No model, dataset, or optimizer is
loaded here, and no new test predictions are generated.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_artifacts import ExperimentRun, sha256_file


SEEDS = (42, 52, 62, 72, 82)
SUBSETS = ("historic_test_complete", "historic_test_strict")
ENDPOINTS = ("accuracy", "macro_f1", "weighted_f1", "negative_f1")
FAMILIES = ("mbert", "xlm_roberta")
AGGREGATES = {
    "mbert": "mbert_final_five_seed_v1",
    "xlm_roberta": "xlm_roberta_final_five_seed_v1",
}
MATRIX = ROOT / "configs/final/multilingual_transformer_comparison_v1.json"
SOURCES = ("HUMIR", "magaza_yorumlari", "random", "tweet-pn", "urun_yorumlari", "wiki")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def verify_manifest(directory: Path, expected_id: str) -> dict[str, dict]:
    status = read_json(directory / "run_status.json")
    manifest = read_json(directory / "artifact_manifest.json")
    require_equal(status["status"], "complete", f"{expected_id} status")
    require_equal(status["run_id"], expected_id, f"{expected_id} status ID")
    require_equal(manifest["run_status"], "complete", f"{expected_id} manifest status")
    require_equal(manifest["run_id"], expected_id, f"{expected_id} manifest ID")
    entries = {}
    for item in manifest["files"]:
        name = item["relative_to_run"]
        if name in entries:
            raise ValueError(f"Duplicate manifest entry: {expected_id}/{name}")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing or escaping artifact: {expected_id}/{name}")
        require_equal(path.stat().st_size, item["size_bytes"], f"{expected_id}/{name} size")
        require_equal(sha256_file(path), item["sha256"], f"{expected_id}/{name} hash")
        entries[name] = item
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="multilingual_paired_seed_diagnostics_v1")
    args = parser.parse_args()
    matrix = read_json(MATRIX)
    require_equal(matrix["status"], "frozen", "comparison matrix status")
    require_equal(matrix["shared_configuration"]["seeds"], list(SEEDS), "frozen seeds")
    models = {item["model_family"]: item for item in matrix["models"]}
    require_equal(set(models), set(FAMILIES), "frozen model families")
    config = {
        "schema_version": "multilingual-paired-seed-diagnostics-v1.0",
        "run_id": args.run_id,
        "analysis_status": "exploratory_not_confirmatory",
        "family_a": "xlm_roberta",
        "family_b": "mbert",
        "delta_direction": "xlm_roberta_minus_mbert",
        "seeds": list(SEEDS),
        "subsets": list(SUBSETS),
        "endpoints": list(ENDPOINTS),
        "source_slices": list(SOURCES),
        "matrix_sha256": sha256_file(MATRIX),
        "input_aggregates": AGGREGATES,
        "uncertainty_unit": "training_seed_on_one_fixed_test_split",
        "inferential_tests": "none; descriptive paired deltas only",
    }
    output_dir = ROOT / "outputs/statistics" / args.run_id
    with ExperimentRun(
        output_dir,
        run_id=args.run_id,
        run_kind="multilingual_paired_seed_diagnostics",
        config=config,
        runtime_environment={"execution_kind": "cpu_only_postprocessing"},
    ) as run:
        per_family = {}
        source_views = {}
        input_rows = []
        for family in FAMILIES:
            aggregate_id = AGGREGATES[family]
            aggregate_dir = ROOT / "outputs/statistics" / aggregate_id
            entries = verify_manifest(aggregate_dir, aggregate_id)
            for required in ("results.json", "input_inventory.csv", "per_seed_metrics.csv", "aggregate_metrics.csv"):
                if required not in entries:
                    raise ValueError(f"Missing aggregate artifact: {aggregate_id}/{required}")
            result = read_json(aggregate_dir / "results.json")
            require_equal(result["config"]["family"], family, f"{family} aggregate family")
            require_equal(result["config"]["matrix_sha256"], config["matrix_sha256"], f"{family} matrix hash")
            require_equal(result["config"]["seeds"], list(SEEDS), f"{family} seeds")
            require_equal(result["config"]["input_experiment_ids"], models[family]["experiment_ids"], f"{family} membership")
            inventory = result["input_inventory"]
            require_equal(len(inventory), len(SEEDS), f"{family} inventory size")
            for seed, item, expected_id in zip(SEEDS, inventory, models[family]["experiment_ids"]):
                require_equal((item["seed"], item["run_id"], item["prediction_rows"]), (seed, expected_id, 88136), f"{family} seed inventory")
                experiment_dir = ROOT / "outputs/experiments" / expected_id
                require_equal(sha256_file(experiment_dir / "artifact_manifest.json"), item["manifest_sha256"], f"{expected_id} input manifest")
                experiment_manifest = read_json(experiment_dir / "artifact_manifest.json")
                metrics_entry = next((entry for entry in experiment_manifest["files"] if entry["relative_to_run"] == "metrics.json"), None)
                if metrics_entry is None:
                    raise ValueError(f"Missing metrics manifest entry: {expected_id}")
                require_equal(sha256_file(experiment_dir / "metrics.json"), metrics_entry["sha256"], f"{expected_id} metrics hash")
                metrics = read_json(experiment_dir / "metrics.json")
                source_views[(family, seed)] = metrics["views"]
                input_rows.append({"family": family, "seed": seed, "run_id": expected_id, "aggregate_id": aggregate_id, "experiment_manifest_sha256": item["manifest_sha256"], "metrics_sha256": metrics_entry["sha256"]})
            indexed = {}
            for row in result["per_seed_metrics"]:
                key = (row["seed"], row["subset_id"], row["endpoint"])
                if key in indexed or row["family"] != family:
                    raise ValueError(f"Duplicate or wrong-family per-seed row: {family}/{key}")
                indexed[key] = float(row["value"])
                source = source_views[(family, row["seed"])]
                if not math.isclose(indexed[key], float(source[row["subset_id"]][row["endpoint"]]), rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Per-seed metric differs from source: {family}/{key}")
            require_equal(len(indexed), len(SEEDS) * len(SUBSETS) * len(ENDPOINTS), f"{family} endpoint count")
            per_family[family] = indexed
            print(f"Accepted {aggregate_id}: {len(inventory)} frozen runs and {len(indexed)} endpoint values")

        paired_rows = []
        summary_rows = []
        for subset in SUBSETS:
            for endpoint in ENDPOINTS:
                deltas = []
                for seed in SEEDS:
                    key = (seed, subset, endpoint)
                    a = per_family["xlm_roberta"][key]
                    b = per_family["mbert"][key]
                    delta = a - b
                    deltas.append(delta)
                    paired_rows.append({"seed": seed, "subset_id": subset, "endpoint": endpoint, "xlm_roberta": a, "mbert": b, "delta_xlm_roberta_minus_mbert": delta})
                summary_rows.append({"subset_id": subset, "endpoint": endpoint, "n_seeds": len(deltas), "mean_delta": statistics.mean(deltas), "sample_sd_delta": statistics.stdev(deltas), "min_delta": min(deltas), "max_delta": max(deltas), "positive_deltas": sum(delta > 0 for delta in deltas)})

        source_rows = []
        for seed in SEEDS:
            for subset in ("complete", "strict"):
                for source in SOURCES:
                    key = f"source_{subset}:{source}"
                    a = source_views[("xlm_roberta", seed)][key]
                    b = source_views[("mbert", seed)][key]
                    require_equal(a["n"], b["n"], f"{key}/seed{seed} support")
                    require_equal(a["averaging_label_ids"], b["averaging_label_ids"], f"{key}/seed{seed} averaging labels")
                    for endpoint in ("accuracy", "macro_f1", "negative_f1", "off_averaging_label_prediction_rate"):
                        if endpoint == "negative_f1" and 1 not in a["averaging_label_ids"]:
                            continue
                        source_rows.append({"seed": seed, "subset_id": key, "source": source, "support": a["n"], "averaging_labels": "|".join(a["averaging_labels"]), "endpoint": endpoint, "xlm_roberta": a[endpoint], "mbert": b[endpoint], "delta_xlm_roberta_minus_mbert": a[endpoint] - b[endpoint]})

        lines = [
            "# Exploratory mBERT/XLM-R matched-seed diagnostics", "",
            "The frozen five seeds are paired by declared seed and evaluated on one fixed Winvoker test split. Positive deltas favor XLM-R. This direct candidate-to-candidate contrast is **not** the protocol's confirmatory BERTurk-versus-finalist family.", "",
            "| Subset | Endpoint | Mean paired delta | Sample SD | Range | Positive seeds |",
            "| :-- | :-- | --: | --: | :-- | --: |",
        ]
        for row in summary_rows:
            lines.append(f"| {row['subset_id']} | {row['endpoint']} | {row['mean_delta']:+.6f} | {row['sample_sd_delta']:.6f} | [{row['min_delta']:+.6f}, {row['max_delta']:+.6f}] | {row['positive_deltas']}/5 |")
        lines += ["", "The strict subset and source-specific results are diagnostic; source Macro F1 averages only labels supported in that source. See `source_diagnostics.csv` for every seed and source. No p-value or independent-dataset confidence claim is made.", ""]
        result = {"schema_version": config["schema_version"], "config": config, "input_inventory": input_rows, "paired_seed_metrics": paired_rows, "paired_summary": summary_rows, "source_diagnostics": source_rows}
        run.write_csv("input_inventory.csv", input_rows)
        run.write_csv("paired_seed_metrics.csv", paired_rows)
        run.write_csv("paired_summary.csv", summary_rows)
        run.write_csv("source_diagnostics.csv", source_rows)
        run.write_json("results.json", result)
        run.write_text("report.md", "\n".join(lines))
        run.write_text("notes.md", "# Notes\n\nExploratory descriptive contrast only; no model loading, training, inference, or confirmatory p-value.\n")
        run.declare_complete(("input_inventory.csv", "paired_seed_metrics.csv", "paired_summary.csv", "source_diagnostics.csv", "results.json", "report.md", "notes.md"))
    print(f"Run completed: {args.run_id}")


if __name__ == "__main__":
    main()
