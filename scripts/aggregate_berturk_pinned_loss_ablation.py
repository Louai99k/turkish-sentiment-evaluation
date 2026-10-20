"""CPU-only additive post-review loss aggregate from the accepted ten-input audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.berturk_replacement_execution_contract import manifest_gate, read, require, sha
from scripts.aggregate_berturk_loss_ablation_v2 import exact_paired_seed_bootstrap, exact_sign_flip_test
from src.experiment_artifacts import ExperimentRun

AUDIT = ROOT / "outputs/audits/professor_round1_berturk_pinned_inputs_audit_v1"
RUN_ID = "professor_round1_berturk_pinned_loss_ablation_v1"
PINS = {
    "report.json": "f399a76e2fa56824cfac56a64f1c810d5ba6954deccfde4874d03a5ff0b912e3",
    "per_seed_metrics.csv": "5b9876cb8fde2c9f49f46bdc31908278f9995e5a916855ae2694af0cb53ce6e3",
    "roster.json": "826b7965dbabbcd66364daaba6a91532b8859052ca33b5035889e40115804120",
    "input_inventory.json": "93c16be9d548dd3b77d124369b6f3307a1ea8f085d1721631b671dd63cfafef7",
}
SEEDS = (42, 52, 62, 72, 82)
VARIANTS = ("weighted", "unweighted")
SUBSETS = ("historic_test_complete", "historic_test_strict")
ENDPOINTS = ("accuracy", "macro_f1", "weighted_f1", "negative_precision", "negative_recall", "negative_f1")


def accepted_inputs():
    manifest_gate(AUDIT)
    for name, pin in PINS.items():
        require(sha(AUDIT / name) == pin, f"Accepted input-audit pin changed: {name}")
    report = read(AUDIT / "report.json")
    require(report["status"] == "ten_fit_inputs_passed_pending_review" and
            report["runs"] == 10 and report["metric_views_checked"] == 20 and
            report["prediction_rows_checked"] == 881360 and
            report["checkpoint_files_hash_verified"] == 10, "Accepted audit scope differs")
    preservation = read(AUDIT / "preservation_check.json")
    require(preservation["inputs_unchanged"] and preservation["cache_files_unchanged"], "Audit preservation failed")
    before = read(AUDIT / "input_inventory.json")
    for path in [*AUDIT.iterdir(), Path(__file__)]:
        if path.is_file():
            before[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, pin in before.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT.resolve()) and sha(path) == pin, f"Audited input changed: {name}")
    cache = read(ROOT / "outputs/audits/professor_round1_berturk_replacement_plan_v1/cache_inventory.json")["files"]
    for row in cache:
        require(sha(Path(row["path"])) == row["sha256"], "Pinned cache changed")
    return before, cache


def aggregate(frame, roster):
    require(len(frame) == 20 and not frame.duplicated(["variant", "seed", "subset_id"]).any(), "Wrong metric grid")
    require(set(frame["subset_id"]) == set(SUBSETS) and set(frame["variant"]) == set(VARIANTS), "Wrong subsets/variants")
    mapping = {(r["variant"], r["seed"]): r for r in roster}
    require(len(roster) == len(mapping) == 10 and
            set(mapping) == {(v, s) for v in VARIANTS for s in SEEDS}, "Wrong roster")
    for row in frame.to_dict("records"):
        item = mapping[(row["variant"], row["seed"])]
        require(row["run_id"] == item["run_id"] and row["origin"] == item["origin"] and
                row["n"] == (88136 if row["subset_id"] == SUBSETS[0] else 87632), "Metric identity/support differs")
    require(np.isfinite(frame[list(ENDPOINTS)].to_numpy(dtype=float)).all(), "Nonfinite metric")
    aggregates, deltas, effects = [], [], []
    for subset in SUBSETS:
        by_variant = {}
        for variant in VARIANTS:
            selected = frame.loc[(frame["variant"] == variant) & (frame["subset_id"] == subset)].sort_values("seed")
            require(selected["seed"].tolist() == list(SEEDS), "Incomplete seed set")
            by_variant[variant] = selected
            for endpoint in ENDPOINTS:
                values = selected[endpoint].to_numpy(dtype=float)
                aggregates.append({"subset_id": subset, "variant": variant, "endpoint": endpoint,
                    "n_seeds": 5, "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()), "maximum": float(values.max())})
        for endpoint in ENDPOINTS:
            weighted = by_variant["weighted"][endpoint].to_numpy(dtype=float)
            standard = by_variant["unweighted"][endpoint].to_numpy(dtype=float)
            difference = weighted - standard
            bootstrap, test = exact_paired_seed_bootstrap(difference), exact_sign_flip_test(difference)
            for i, seed in enumerate(SEEDS):
                deltas.append({"subset_id": subset, "seed": seed, "endpoint": endpoint,
                    "weighted_run_id": mapping[("weighted", seed)]["run_id"],
                    "unweighted_run_id": mapping[("unweighted", seed)]["run_id"],
                    "weighted_value": float(weighted[i]), "unweighted_value": float(standard[i]),
                    "delta_weighted_minus_unweighted": float(difference[i])})
            effects.append({"subset_id": subset, "endpoint": endpoint, "n_seed_pairs": 5,
                "delta_direction": "weighted_minus_unweighted", "delta_mean": float(difference.mean()),
                "delta_sample_sd": float(difference.std(ddof=1)), "delta_median": float(np.median(difference)),
                "delta_min": float(difference.min()), "delta_max": float(difference.max()),
                "weighted_wins": int((difference > 0).sum()), "ties": int((difference == 0).sum()),
                "unweighted_wins": int((difference < 0).sum()),
                "bootstrap_resamples": bootstrap["enumerated_resamples"],
                "bootstrap_ci_low": bootstrap["ci_low"], "bootstrap_ci_high": bootstrap["ci_high"],
                "sign_flip_assignments": test["enumerated_assignments"],
                "sign_flip_extreme_assignments": test["extreme_assignments"],
                "sign_flip_p_value": test["p_value"], "minimum_two_sided_p_value": test["minimum_two_sided_p_value"]})
    return aggregates, deltas, effects


def render_report(aggregates, effects):
    lines = ["# Pinned BERTurk loss ablation — post-review exploratory evidence", "",
        "Six replacements plus four accepted pinned standards; historical aggregate and original confirmatory analysis preserved.",
        "All ten record the immutable model revision, but six new generic-runner fits and four earlier fits retain distinct environments/metadata eras; no homogeneous or bitwise historical replay claim.",
        "Seeds 42/52/62/72/82, fixed benchmark; delta = weighted minus standard CE. Macro F1 primary interpretation, Negative F1 minority endpoint, Weighted F1 continuity endpoint.",
        "Sample SD uses ddof=1. Paired-seed percentile bootstrap enumerates all 3125 resamples (95% interval); two-sided sign flips enumerate 32 assignments (minimum p=.0625).",
        "These exploratory p-values are unadjusted, not the separate competitor confirmatory family. Nonsignificance is not equivalence; report every endpoint and subset, not only favorable results.",
        "Pending independent aggregate audit: no manuscript/claim refresh yet.", "",
        "| Subset | Loss | Endpoint | Mean | Sample SD |", "| --- | --- | --- | ---: | ---: |"]
    for row in aggregates:
        lines.append(f"| {row['subset_id']} | {row['variant']} | {row['endpoint']} | {row['mean']:.8f} | {row['sample_sd']:.8f} |")
    lines += ["", "| Subset | Endpoint | Mean delta | 95% paired-seed interval | Unadjusted sign-flip p |", "| --- | --- | ---: | --- | ---: |"]
    for row in effects:
        lines.append(f"| {row['subset_id']} | {row['endpoint']} | {row['delta_mean']:.8f} | [{row['bootstrap_ci_low']:.8f}, {row['bootstrap_ci_high']:.8f}] | {row['sign_flip_p_value']:.6f} |")
    return "\n".join(lines) + "\n"


def main():
    output = ROOT / "outputs/statistics" / RUN_ID
    require(not output.exists(), "Refusing existing aggregate output")
    config = {"analysis_scope": "post_review_exploratory_pinned_loss_ablation",
        "input_audit_id": AUDIT.name, "input_pins": PINS, "seeds": list(SEEDS),
        "variants": list(VARIANTS), "subsets": list(SUBSETS), "endpoints": list(ENDPOINTS),
        "delta_direction": "weighted_minus_unweighted", "sd_ddof": 1,
        "bootstrap": "exact_3125_paired_seed_percentile_95pct",
        "test": "exact_32_two_sided_paired_seed_sign_flips_unadjusted_exploratory",
        "training": False, "inference": False, "publication_authorized": False}
    with ExperimentRun(output, run_id=RUN_ID, run_kind="cpu_post_review_pinned_loss_aggregate", config=config) as run:
        print("Checking accepted ten-run audit, protected inputs and cached pins", flush=True)
        before, cache = accepted_inputs()
        run.write_json("input_inventory.json", before)
        try:
            frame = pd.read_csv(AUDIT / "per_seed_metrics.csv", keep_default_na=False, float_precision="round_trip")
            roster = read(AUDIT / "roster.json")
            aggregates, deltas, effects = aggregate(frame, roster)
            require("torch" not in sys.modules, "Unexpected torch import")
            run.write_json("roster.json", roster)
            run.write_csv("per_seed_metrics.csv", frame)
            run.write_csv("aggregate_metrics.csv", pd.DataFrame(aggregates))
            run.write_csv("paired_seed_deltas.csv", pd.DataFrame(deltas))
            run.write_csv("paired_seed_summary.csv", pd.DataFrame(effects))
            run.write_json("results.json", {"status": "aggregate_computed_pending_independent_audit",
                "config": config, "roster": roster, "aggregate_metrics": aggregates,
                "paired_seed_summary": effects, "checkpoint_deserialized": False,
                "next": "independent aggregate audit then separate loss-labelled post-review competitor statistics"})
            run.write_text("report.md", render_report(aggregates, effects))
        finally:
            unchanged = all(sha(ROOT / name) == pin for name, pin in before.items())
            cached = all(sha(Path(r["path"])) == r["sha256"] for r in cache)
            run.write_json("preservation_check.json", {"inputs_unchanged": unchanged,
                "checked_files": len(before), "cache_files_unchanged": cached, "cache_files_checked": len(cache)})
            require(unchanged and cached, "Aggregate input/cache preservation failed")
        run.declare_complete(["input_inventory.json", "roster.json", "per_seed_metrics.csv",
            "aggregate_metrics.csv", "paired_seed_deltas.csv", "paired_seed_summary.csv",
            "results.json", "report.md", "preservation_check.json"])


if __name__ == "__main__":
    main()
