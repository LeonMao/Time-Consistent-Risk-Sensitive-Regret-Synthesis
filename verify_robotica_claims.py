from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "06_DATA_AND_RESULTS"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def truth(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def pick(table: list[dict[str, str]], **criteria: object) -> dict[str, str]:
    matches = [
        row
        for row in table
        if all(row[key] == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {criteria}, got {len(matches)}")
    return matches[0]


def verify_time_consistency() -> str:
    table = rows("stage3_3_p2_time_consistency_summary.csv")
    static = pick(table, criterion="static_precommitment_CVaR_regret")
    nested = pick(table, criterion="nested_dynamic_CVaR_regret")
    assert (static["planned_action_at_rare_H"], static["action_after_resolving_same_criterion_at_H"]) == ("a", "b")
    assert (nested["planned_action_at_rare_H"], nested["action_after_resolving_same_criterion_at_H"]) == ("b", "b")
    close(number(static, "mismatch_probability"), 0.05)
    close(number(nested, "mismatch_probability"), 0.0)
    return "static CVaR replans with probability 0.05; nested risk is time-consistent"


def verify_full_policy_exactness() -> str:
    table = rows("stage3_3_p0_full_policy_exactness.csv")
    assert len(table) == 21
    assert all(truth(row["full_computed_policy_map_equal"]) for row in table)
    assert all(number(row, "different_action_entries") == 0 for row in table)
    maximum = max(number(row, "value_abs_error") for row in table)
    assert maximum <= 8e-15
    return f"21/21 explicit/factored policy maps agree; max value error={maximum:.3e}"


def verify_static_nested_mismatch() -> str:
    table = rows("stage3_5_static_nested_mismatch_summary.csv")
    random_rows = [row for row in table if truth(row["non_hand_constructed"])]
    assert len(random_rows) == 720
    mismatch = sum(number(row, "static_action_mismatch_state_count") > 0 for row in random_rows)
    reachable = sum(truth(row["reachable_policy_difference"]) for row in random_rows)
    root = sum(truth(row["root_action_difference"]) for row in random_rows)
    max_improvement = max(number(row, "max_static_conditional_cvar_improvement") for row in random_rows)
    assert (mismatch, reachable, root) == (97, 307, 224)
    close(max_improvement, 2.8028362409)
    return "97/720 randomized instances exhibit a static replan mismatch; max improvement=2.803"


def verify_alpha_tradeoff() -> str:
    table = rows("stage3_3_p0_exact_alpha_m8_summary.csv")
    a0 = pick(table, planning_alpha="0.0")
    a5 = pick(table, planning_alpha="0.5")
    close(number(a0, "exact_mean_regret_mean"), 12.12157018160253)
    close(number(a0, "exact_cvar95_regret_mean"), 30.864935783142442)
    close(number(a5, "exact_mean_regret_mean"), 23.956052947517527)
    close(number(a5, "exact_cvar95_regret_mean"), 26.90684060616855)
    reduction = 1.0 - number(a5, "exact_cvar95_regret_mean") / number(a0, "exact_cvar95_regret_mean")
    close(reduction, 0.12823921633220092)
    assert all(truth(row["all_worlds_satisfied"]) for row in table)
    return f"alpha=0.5 lowers CVaR95 regret by {100 * reduction:.2f}% while preserving all-world satisfaction"


def verify_minimax_endpoint() -> str:
    thresholds = rows("stage3_3_p1_minimax_thresholds.csv")
    hub8 = sorted(number(row, "sufficient_alpha_threshold") for row in thresholds if row["family"] == "hub_m8")
    hub10 = sorted(number(row, "sufficient_alpha_threshold") for row in thresholds if row["family"] == "hub_m10")
    intel = [number(row, "sufficient_alpha_threshold") for row in thresholds if row["family"] == "intel_scLTL"]
    assert (len(hub8), len(hub10), len(intel)) == (10, 10, 1)
    close(median(hub8), 0.9756788309271641)
    close(min(hub8), 0.9627820503925686)
    close(max(hub8), 0.9826303630203939)
    close(median(hub10), 0.988249040898129)
    close(intel[0], 0.9883980208)

    plateau = rows("stage3_3_p1_hub_m8_plateau_10seeds.csv")
    first_hits = []
    for seed in range(10):
        candidates = [number(row, "alpha") for row in plateau if row["seed"] == str(seed) and truth(row["at_minimax_value"])]
        first_hits.append(min(candidates))
    assert min(first_hits) >= 0.65 and max(first_hits) <= 0.80
    close(median(first_hits), 0.75)
    return "high-alpha nested risk reaches the minimax plateau; hub-m8 median first grid hit=0.75"


def verify_disagreement() -> str:
    table = rows("stage3_3_p2_3_disagreement_by_alpha.csv")
    assert len(table) == 4 and sum(int(row["instances"]) for row in table) == 200
    behavior = sum(int(row["instances"]) * number(row, "behavior_disagreement_rate") for row in table)
    root = sum(int(row["instances"]) * number(row, "root_disagreement_rate") for row in table)
    close(behavior, 27.0)
    close(root, 22.0)
    a25 = pick(table, alpha="0.25")
    close(number(a25, "behavior_disagreement_rate"), 0.32)
    close(number(a25, "root_disagreement_rate"), 0.26)
    close(number(a25, "delta_expected_regret_on_disagreement"), -1.560164619158618)
    close(number(a25, "delta_worst_cost_on_disagreement"), 1.643354430533233)
    return "cost-risk and regret-risk disagree behaviorally in 27/200 cases and at the root in 22/200"


def verify_controlled_scaling() -> str:
    table = rows("stage3_5_controlled_timing_statistics.csv")
    primary = [row for row in table if row["aggregate_role"] == "paired_five_seed"]
    assert len(primary) == 4
    speedups = [number(row, "paired_speedup_median_x") for row in primary]
    close(min(speedups), 2.11514980395)
    close(max(speedups), 33.7009184752)
    hub10 = pick(table, family="hub", m="10")
    assert int(float(hub10["factored_oracle_calls_median"])) == 122
    assert int(float(hub10["explicit_oracle_calls_median"])) == 1024
    assert int(float(hub10["explicit_posterior_entries_median"])) == 11264
    assert int(float(hub10["factored_probability_entries_median"])) == 662
    layered10 = pick(table, family="layered", m="10")
    assert int(float(layered10["factored_oracle_calls_median"])) == 1024
    assert int(float(layered10["explicit_oracle_calls_median"])) == 1024
    return "four controlled configurations show median speedups from 2.12x to 33.70x"


def verify_horizon_and_rank() -> str:
    horizon = sorted(rows("stage1_7_summary_horizon.csv"), key=lambda row: int(row["H"]))
    values = [number(row, "dynamic_objective_mean") for row in horizon]
    assert len(values) == 5 and all(a + 1e-12 >= b for a, b in zip(values, values[1:]))
    close(values[0], 26.052477439290794)
    close(values[-1], 4.789470948358791)
    assert int(float(horizon[0]["agent_states_mean"])) == 18
    assert int(float(horizon[-1]["agent_states_mean"])) == 7218

    rank_rows = rows("stage3_5_minimal_robust_ranks.csv")
    intel_rank = pick(rank_rows, family="intel_lab_topological")
    assert (int(intel_rank["minimal_robust_rank"]), int(intel_rank["supported_world_count"]), int(intel_rank["dfa_state_count"])) == (9, 32, 4)
    intel_h = rows("stage3_5_intel_horizon_sensitivity.csv")
    assert len(intel_h) == 25 and all(truth(row["all_worlds_satisfied"]) for row in intel_h)
    alpha0 = [row for row in intel_h if row["alpha"] == "0"]
    assert {int(row["first_joint_stable_h_through_max"]) for row in alpha0} == {11}
    return "objective is nonincreasing with horizon; Intel robust rank=9 and alpha=0 stabilizes at H=11"


def verify_prior_robustness() -> str:
    table = rows("stage3_5_prior_robustness.csv")
    assert len(table) == 100
    weight = [row for row in table if row["scenario_class"] == "prior_weight_error"]
    missing = [row for row in table if row["scenario_class"] == "missing_support_error"]
    assert len(weight) == 70 and sum(truth(row["root_action_changed"]) for row in weight) == 10
    assert len(missing) == 25
    undefined = sum(row["nominal_policy_evaluation_status"] != "complete" for row in missing)
    lost = sum(not truth(row["hard_guarantee_retained_by_nominal_policy"]) for row in missing)
    safe = sum(truth(row["resynthesized_all_true_worlds_satisfied"]) for row in missing)
    assert (undefined, lost, safe) == (20, 20, 25)
    return "weight error changes 10/70 roots; missing support breaks 20/25 nominal guarantees; resynthesis is safe in 25/25"


def verify_baselines() -> str:
    summary = rows("stage3_3_p0_baseline_comparison_summary.csv")
    expected = pick(summary, family="random_hub_m8", method="Expected cost / regret")
    proposed = pick(summary, family="random_hub_m8", method="Proposed dynamic CVaR regret a=0.5")
    close(number(expected, "mean_regret"), 12.12157018160253)
    close(number(expected, "cvar95_regret"), 30.864935783142442)
    close(number(proposed, "mean_regret"), 23.956052947517527)
    close(number(proposed, "cvar95_regret"), 26.90684060616855)

    common = rows("stage3_5_common_objective_baselines.csv")
    def common_mean(family: str, method: str) -> float:
        subset = [number(row, "common_proposed_nested_dynamic_regret") for row in common if row["family"] == family and row["method"] == method]
        if not subset:
            raise AssertionError((family, method))
        return sum(subset) / len(subset)

    close(common_mean("random_hub_m8", "Expected cost / regret"), 30.23556260285)
    close(common_mean("random_hub_m8", "Proposed dynamic CVaR regret a=0.5"), 26.59175000271)
    close(common_mean("intel_map_scLTL", "Expected cost / regret"), 5.66566451222)
    close(common_mean("intel_map_scLTL", "Proposed dynamic CVaR regret a=0.5"), 5.64477689287)
    return "baseline rows reproduce the native-metric and common-objective comparisons"


def verify_figures_and_paper_lock() -> str:
    expected_figures = {
        "fig_problem_setting.png",
        "fig_method_framework.png",
        "fig_method_factored_solver.png",
        "fig_stage3_5_pkwts_time_consistency_grayscale.png",
        "fig_stage3_3_exact_alpha_tradeoff.png",
        "fig_stage3_3_p1_minimax_plateau.png",
        "fig_stage1_7_horizon_value.png",
        "fig_stage3_5_intel_lab_topology_grayscale.png",
        "fig_stage3_5_layered_benchmark_grayscale.png",
        "fig_stage3_5_controlled_timing_statistics_grayscale.png",
    }
    actual = {path.name for path in (ROOT / "01_FIGURES").glob("*.png")}
    assert actual == expected_figures
    assert not list(ROOT.rglob("generate_method_figures.py"))

    lock = json.loads((ROOT / "PAPER_VERSION.json").read_text(encoding="utf-8"))
    assert lock["paper_included_in_release"] is False
    assert len(lock["source_sha256"]) == 64 and len(lock["compiled_pdf_sha256"]) == 64
    manifest = json.loads((ROOT / "claim_evidence_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["claims"]) == 11
    return "10 paper figures are present; the 3 author-drawn method figures have no generator; paper is hash-locked but excluded"


CHECKS = [
    ("C01", verify_time_consistency),
    ("C02", verify_full_policy_exactness),
    ("C03", verify_static_nested_mismatch),
    ("C04", verify_alpha_tradeoff),
    ("C05", verify_minimax_endpoint),
    ("C06", verify_disagreement),
    ("C07", verify_controlled_scaling),
    ("C08", verify_horizon_and_rank),
    ("C09", verify_prior_robustness),
    ("C10", verify_baselines),
    ("C11", verify_figures_and_paper_lock),
]


def main() -> int:
    failures: list[str] = []
    for claim_id, check in CHECKS:
        try:
            detail = check()
            print(f"[PASS:{claim_id}] {detail}")
        except Exception as exc:  # every failed claim must be reported
            failures.append(f"{claim_id}: {exc}")
            print(f"[FAIL:{claim_id}] {exc}")
    if failures:
        print(f"ROBOTICA CLAIM VERIFICATION: FAIL ({len(failures)} claims)")
        return 1
    print(f"ROBOTICA CLAIM VERIFICATION: PASS ({len(CHECKS)} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
