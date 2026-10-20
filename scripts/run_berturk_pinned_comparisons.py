"""CPU fixed-fit post-review comparisons, delegating only to unchanged resampling functions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts import significance_tests as engine
from scripts.berturk_replacement_execution_contract import manifest_gate, read, require, sha
from src.experiment_artifacts import ExperimentRun

METHOD = ROOT / "configs/statistics/professor_round1_pinned_comparisons_method_v1.json"
FAMILIES = {"weighted": "professor_round1_pinned_weighted_ce_comparisons_v2",
            "unweighted": "professor_round1_pinned_standard_ce_comparisons_v1"}
AUDIT = ROOT / "outputs/audits/professor_round1_berturk_pinned_loss_ablation_audit_v1"
ORIGINAL = ROOT / "outputs/statistics/final_confirmatory_comparisons_v1/results.json"


def recorded_training_seed(config, checkpoint, exp_id):
    """Check all recorded seed fields, never infer a seed from a directory name."""
    require(config.get("run_id") == exp_id, f"Config run ID differs: {exp_id}")
    config_seeds = [config[key] for key in ("seed", "training_seed") if key in config]
    identity = config.get("run_identity", {})
    if "seed" in identity:
        require(identity.get("run_id") == exp_id, f"Nested run ID differs: {exp_id}")
        config_seeds.append(identity["seed"])
    checkpoint_seeds = []
    if "training_seed" in checkpoint.get("metadata", {}):
        checkpoint_seeds.append(checkpoint["metadata"]["training_seed"])
    if "seed" in checkpoint.get("run_identity", {}):
        require(checkpoint["run_identity"] == identity, f"Checkpoint run identity differs: {exp_id}")
        checkpoint_seeds.append(checkpoint["run_identity"]["seed"])
    require(config_seeds and checkpoint_seeds and
            all(type(value) is int and value == 42 for value in [*config_seeds, *checkpoint_seeds]),
            f"Missing/conflicting training-seed42 records: {exp_id}")
    return {"run_id": exp_id, "config_seed_fields": config_seeds,
            "checkpoint_seed_fields": checkpoint_seeds, "training_seed": 42}


def gate(loss_mode):
    family_path = ROOT / "configs/statistics" / f"{FAMILIES[loss_mode]}.json"
    family, method = read(family_path), read(METHOD)
    require(family["analysis_id"] == FAMILIES[loss_mode] and family["loss_mode"] == loss_mode and
            family["anchor"] == f"rev1_berturk_winvoker_canonical_len128_seed42_{loss_mode}_pinned" and
            family["method_contract"] == METHOD.relative_to(ROOT).as_posix() and
            family["analysis_scope"] == method["analysis_scope"] == "post_review_pinned_replacement" and
            family["confirmatory"] is method["confirmatory"] is False and method["seed_series"] == [],
            "Wrong post-review family")
    require(sha(ROOT / "scripts/significance_tests.py") == method["engine_sha256"] ==
            "c773baf9264d261ee90f993a6364dcf414d17f69f1bce6a168b8b198de9bec5d", "Original engine changed")
    require(sha(ORIGINAL) == method["original_results_sha256"] ==
            "eae3785a9da1c78b59cd5e8ffbbfca84f781bb34185f94bd2d6b0091dec2cc18", "Original results changed")
    manifest_gate(AUDIT)
    require(sha(AUDIT / "report.json") == method["accepted_aggregate_audit_report_sha256"] ==
            "bdf8cf36a08b8d9034b0f32445c83b746cd8f76284b33509cb55f9323846a038" and
            sha(AUDIT / "input_inventory.json") == method["accepted_aggregate_audit_inventory_sha256"] ==
            "5c89fff900a0231113ee921586d68bdf49718e109530fb248dbc3ef8cc67e2a6", "Accepted audit changed")
    report, preservation = read(AUDIT / "report.json"), read(AUDIT / "preservation_check.json")
    require(report["status"] == "aggregate_arithmetic_passed_pending_review" and
            all(c["passed"] for c in report["checks"]) and preservation["inputs_unchanged"] and
            preservation["cache_files_unchanged"], "Prerequisite audit incomplete")
    before = read(AUDIT / "input_inventory.json")
    for path in [*AUDIT.iterdir(), METHOD, Path(__file__), ROOT / "scripts/significance_tests.py",
                 ROOT / "configs/statistics/professor_round1_pinned_weighted_ce_comparisons_v1.json",
                 *[ROOT / "configs/statistics" / f"{name}.json" for name in FAMILIES.values()]]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha(path)
    if loss_mode == "weighted":
        previous = ROOT / "outputs/statistics/professor_round1_pinned_weighted_ce_comparisons_v1"
        previous_status = read(previous / "run_status.json")
        require(family["supersedes_failed_analysis"] == previous.name and
                previous_status["status"] == "failed" and
                previous_status["error"] == {"type": "KeyError", "message": "'seed'"} and
                not (previous / "results.json").exists(), "Prior failure differs; inspect before retry")
        for path in previous.rglob("*"):
            if path.is_file():
                before[path.relative_to(ROOT).as_posix()] = sha(path)
    original = read(ORIGINAL)
    for key in ("subset", "stratify", "metrics", "n_bootstrap", "n_randomizations",
                "randomization_batch_size", "confidence_level", "base_seed"):
        require(method[key] == original["config"][key], f"Original method differs: {key}")
    competitors = method["competitors"]
    require(len(competitors) == 5 and [c["exp_b"] for c in competitors] ==
            [c["exp_b"] for c in original["comparisons"]], "Original competitors/order changed")
    for competitor, old in zip(competitors, original["comparisons"]):
        require(competitor["predictions_sha256"] == old["input_predictions"]["exp_b_sha256"], "Competitor pin differs")
    expected_inputs = [(family["anchor"], family["anchor_predictions_sha256"])] + [
        (c["exp_b"], c["predictions_sha256"]) for c in competitors]
    seed_records = []
    for exp_id, prediction_pin in expected_inputs:
        directory = ROOT / "outputs/experiments" / exp_id
        status, config = read(directory / "run_status.json"), read(directory / "config.json")
        require(status["status"] == "complete" and status["error"] is None,
                "Incomplete input")
        seed_records.append(recorded_training_seed(config, read(directory / "checkpoint_metadata.json"), exp_id))
        require(sha(directory / "predictions.csv") == prediction_pin, f"Prediction pin differs: {exp_id}")
        for name in ("predictions.csv", "config.json", "run_status.json", "artifact_manifest.json", "checkpoint_metadata.json"):
            path = directory / name
            before[path.relative_to(ROOT).as_posix()] = sha(path)
    anchor = ROOT / "outputs/experiments" / family["anchor"]
    require(read(anchor / "config.json")["loss_mode"] == loss_mode and
            before[(anchor / "predictions.csv").relative_to(ROOT).as_posix()] ==
            read(AUDIT / "input_inventory.json")[(anchor / "predictions.csv").relative_to(ROOT).as_posix()] and
            sha(anchor / "checkpoints/best_checkpoint.pt") == family["anchor_checkpoint_sha256"], "Pinned anchor differs")
    for name, pin in before.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha(path) == pin, f"Protected input changed: {name}")
    cache = read(ROOT / "outputs/audits/professor_round1_berturk_replacement_plan_v1/cache_inventory.json")["files"]
    for row in cache:
        require(sha(Path(row["path"])) == row["sha256"], "Pinned cache changed")
    return family, method, family_path, before, cache, seed_records


def calculate(family, method):
    lineage = engine.load_lineage(engine.DEFAULT_LINEAGE_DIR)
    mask = engine.subset_mask(lineage, method["subset"])
    require(lineage.file_sha256 == method["lineage_mask_sha256"] ==
            "635f28535111efec1baeae4b9ba2e3f28b7dde8faa6240cf4b9afdad86de5999" and
            len(mask) == 88136 and int(mask.sum()) == 87632, "Strict lineage differs")
    positions = np.flatnonzero(mask)
    hashes = tuple(lineage.cleaned_text_sha256[i] for i in positions)
    sources = tuple(lineage.source[i] for i in positions)
    a = engine.load_predictions(family["anchor"], lineage)
    comparisons, endpoints, strata_rows = [], [], []
    for index, item in enumerate(method["competitors"], 1):
        print(f"[{index}/5] {family['loss_mode']} seed42 vs {item['name']}", flush=True)
        b = engine.load_predictions(item["exp_b"], lineage)
        require(a.file_sha256 == family["anchor_predictions_sha256"] and
                b.file_sha256 == item["predictions_sha256"] and np.array_equal(a.y_true, b.y_true), "Prediction alignment differs")
        y, pa, pb = a.y_true[mask], a.y_pred[mask], b.y_pred[mask]
        design = engine.build_group_design(y, pa, pb, hashes, sources, method["stratify"])
        boot_seed = engine.stable_seed(method["base_seed"], a.experiment_id, b.experiment_id, "bootstrap")
        swap_seed = engine.stable_seed(method["base_seed"], a.experiment_id, b.experiment_id, "randomization")
        boot = engine.paired_cluster_bootstrap(design, method["metrics"], method["n_bootstrap"], method["confidence_level"], boot_seed)
        swaps = engine.paired_group_randomization(design, method["metrics"], method["n_randomizations"], swap_seed, method["randomization_batch_size"])
        cid = f"{family['loss_mode']}_berturk_vs_{index:02d}"
        comparisons.append({"comparison_id": cid, "exp_a": a.experiment_id, "exp_b": b.experiment_id,
            "competitor": item["name"], "n_items": int(mask.sum()),
            "input_predictions": {"exp_a_sha256": a.file_sha256, "exp_b_sha256": b.file_sha256},
            "group_design": design["diagnostics"], "bootstrap": boot, "randomization": swaps,
            "mcnemar": engine.mcnemar_exact(y, pa, pb)})
        strata_rows.extend({"comparison_id": cid, **s} for s in design["diagnostics"]["strata"])
        for metric in method["metrics"]:
            effect, test = boot["endpoints"][metric], swaps["endpoints"][metric]
            endpoints.append({"analysis_scope": method["analysis_scope"], "confirmatory": False,
                "loss_mode": family["loss_mode"], "subset": method["subset"], "comparison_id": cid,
                "exp_a": a.experiment_id, "exp_b": b.experiment_id, "competitor": item["name"],
                "metric": metric, "endpoint_role": engine.endpoint_role(metric), **effect,
                "confidence_level": method["confidence_level"], "n_bootstrap": method["n_bootstrap"],
                "bootstrap_seed": boot_seed, "randomization_seed": swap_seed,
                "n_randomizations": method["n_randomizations"], "extreme_randomizations": test["extreme_randomizations"],
                "p_value_raw": test["p_value_raw"], "p_value_resolution": test["p_value_resolution"],
                "holm_family": f"{family['analysis_id']}:{metric}", "holm_family_size": 5})
    for metric in method["metrics"]:
        selected = [r for r in endpoints if r["metric"] == metric]
        require(len(selected) == 5, "Holm family incomplete")
        for row, adjusted in zip(selected, engine.holm_adjust([r["p_value_raw"] for r in selected])):
            row["p_value_holm"] = adjusted
    require(len(endpoints) == 15 and "torch" not in sys.modules, "Wrong endpoint count/model import")
    return comparisons, endpoints, strata_rows


def report_text(family, rows):
    lines = [f"# Pinned {family['loss_mode']} CE BERTurk comparisons — post-review", "",
        "Five fixed seed42 competitors, strict87632 rows. Not original confirmatory evidence or blind preregistration.",
        "5000 source-label-stratified cleaned-text-group bootstraps,95% percentile intervals;5000 group swaps with plus-one p correction.",
        "Holm5 separately within this loss-anchor/endpoint family. Fixed-fit conditional inference, not training-seed variability or equivalence.",
        "Base42 seed derivation is unchanged; new experiment IDs produce new deterministic bootstrap/randomization seeds, not identical historical draws.",
        "Both loss families/all30 endpoints must be reported after their audits. McNemar is a row-level diagnostic, not a grouped primary test.",
        "Pending separate audit: do not update manuscript statistical claims yet.", "",
        "| Competitor | Endpoint | Delta | 95% interval | Raw p | Holm p |",
        "| --- | --- | ---: | --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['competitor']} | {r['metric']} | {r['delta_a_minus_b']:+.8f} | [{r['ci_low']:+.8f}, {r['ci_high']:+.8f}] | {r['p_value_raw']:.8f} | {r['p_value_holm']:.8f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-mode", required=True, choices=tuple(FAMILIES))
    args = parser.parse_args()
    output = ROOT / "outputs/statistics" / FAMILIES[args.loss_mode]
    require(not output.exists(), "Refusing existing statistics output")
    with ExperimentRun(output, run_id=FAMILIES[args.loss_mode], run_kind="cpu_pinned_post_review_statistics",
                       config={"loss_mode": args.loss_mode, "confirmatory": False}) as run:
        print("Checking frozen method, accepted audits and preserved inputs", flush=True)
        family, method, family_path, before, cache, seed_records = gate(args.loss_mode)
        config = {"family": family, "method": method, "family_path": family_path.relative_to(ROOT).as_posix(),
            "family_sha256": sha(family_path), "method_sha256": sha(METHOD),
            "runner_sha256": sha(Path(__file__)), "training": False, "inference": False}
        run.write_json("config.json", config)
        run.write_json("input_inventory.json", before)
        run.write_json("training_seed_records.json", seed_records)
        try:
            comparisons, endpoints, strata = calculate(family, method)
            run.write_json("results.json", {"status": "post_review_statistics_computed_pending_audit",
                "config": config, "comparisons": comparisons, "summary_rows": endpoints,
                "seed_level": {"per_seed_rows": [], "summary_rows": []}, "confirmatory": False,
                "required_audit_id": family["required_audit_id"]})
            run.write_csv("endpoint_results.csv", pd.DataFrame(endpoints))
            run.write_csv("group_strata.csv", pd.DataFrame(strata))
            run.write_text("report.md", report_text(family, endpoints))
        finally:
            unchanged = all(sha(ROOT / name) == pin for name, pin in before.items())
            cached = all(sha(Path(r["path"])) == r["sha256"] for r in cache)
            run.write_json("preservation_check.json", {"inputs_unchanged": unchanged,
                "checked_files": len(before), "cache_files_unchanged": cached, "cache_files_checked": len(cache)})
            require(unchanged and cached, "Statistics input/cache preservation failed")
        run.declare_complete(["input_inventory.json", "training_seed_records.json", "results.json", "endpoint_results.csv",
                              "group_strata.csv", "report.md", "preservation_check.json"])


if __name__ == "__main__":
    main()
