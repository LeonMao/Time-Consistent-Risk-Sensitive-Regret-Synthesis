from __future__ import annotations

import argparse
import csv
from math import nan
from pathlib import Path
from statistics import mean

from stage3_3_reconstructed_common import (
    action_map,
    exact_policy_metrics,
    simulate_budget_policy,
)
from stage1_7_utils import eventually_goal_dfa, independent_prior, random_multishortcut
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from stage3_3_baseline_solvers import FactoredDynamicCVaRCostSolver


SEEDS = tuple(range(10))
P_OPEN_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
ALPHAS = (0.25, 0.5, 0.75, 0.9)
METRIC_NAMES = (
    "mean_cost", "mean_regret", "cvar95_regret", "worst_regret", "worst_cost"
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def conditional_mean(rows: list[dict], field: str, predicate) -> float:
    selected = [float(row[field]) for row in rows if predicate(row)]
    return mean(selected) if selected else nan


def generate_raw(jitter: float) -> tuple[list[dict], list[dict]]:
    policy_rows: list[dict] = []
    behavior_rows: list[dict] = []
    dfa = eventually_goal_dfa()

    for seed in SEEDS:
        transition_system = random_multishortcut(8, seed)
        for p_open in P_OPEN_VALUES:
            prior = independent_prior(
                transition_system, p_open, seed=seed, jitter=jitter
            )
            explicit_prior = prior.explicit_prior_for_validation()
            for alpha in ALPHAS:
                cost_solver = FactoredDynamicCVaRCostSolver(
                    transition_system, dfa, prior, alpha=alpha, horizon=5
                )
                cost_result = cost_solver.solve_cost()
                regret_solver = FactoredLazyHorizonDynamicCVaRSolver(
                    transition_system, dfa, prior, alpha=alpha, horizon=5
                )
                regret_result = regret_solver.solve()

                cost_metrics = exact_policy_metrics(cost_solver, cost_result.policy)
                regret_metrics = exact_policy_metrics(
                    regret_solver, regret_result.policy
                )
                cost_map = action_map(cost_result.policy)
                regret_map = action_map(regret_result.policy)
                common_keys = set(cost_map) & set(regret_map)
                different_common = sum(
                    cost_map[key] != regret_map[key] for key in common_keys
                )
                full_equal = cost_map == regret_map

                behavior_worlds = 0
                behavior_mass = 0.0
                for world, probability in explicit_prior.items():
                    _, _, cost_actions = simulate_budget_policy(
                        cost_solver, cost_result.policy, world
                    )
                    _, _, regret_actions = simulate_budget_policy(
                        regret_solver, regret_result.policy, world
                    )
                    if cost_actions != regret_actions:
                        behavior_worlds += 1
                        behavior_mass += float(probability)

                cost_root = str(cost_result.policy[(cost_solver.start, 5)].target)
                regret_root = str(
                    regret_result.policy[(regret_solver.start, 5)].target
                )
                shared = {
                    "seed": seed,
                    "p_open": p_open,
                    "alpha": alpha,
                    "cost_root_action": cost_root,
                    "regret_root_action": regret_root,
                    "root_action_disagree": int(cost_root != regret_root),
                }
                metric_values: dict[str, float] = {}
                for name in METRIC_NAMES:
                    metric_values[f"cost_{name}"] = float(cost_metrics[name])
                    metric_values[f"regret_{name}"] = float(regret_metrics[name])
                    metric_values[
                        f"delta_{name}_regret_minus_cost"
                    ] = float(regret_metrics[name]) - float(cost_metrics[name])

                policy_rows.append({
                    "seed": seed,
                    "m": 8,
                    "p_open": p_open,
                    "alpha": alpha,
                    "cost_root_action": cost_root,
                    "regret_root_action": regret_root,
                    "root_action_disagree": int(cost_root != regret_root),
                    "full_policy_map_equal": int(full_equal),
                    "common_policy_keys": len(common_keys),
                    "different_common_actions": different_common,
                    "cost_policy_entries": len(cost_map),
                    "regret_policy_entries": len(regret_map),
                    **metric_values,
                })
                behavior_rows.append({
                    **shared,
                    "behavior_disagree": int(behavior_worlds > 0),
                    "behavior_disagreement_world_fraction": (
                        behavior_worlds / len(explicit_prior)
                    ),
                    "behavior_disagreement_prior_mass": behavior_mass,
                    **metric_values,
                })
    return policy_rows, behavior_rows


def summarize_policy(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for alpha in ALPHAS:
        for p_open in P_OPEN_VALUES:
            group = [
                row for row in rows
                if row["alpha"] == alpha and row["p_open"] == p_open
            ]
            disagreed = lambda row: int(row["full_policy_map_equal"]) == 0
            record = {
                "alpha": alpha,
                "p_open": p_open,
                "instances": len(group),
                "full_policy_disagreement_rate": mean(
                    int(not int(row["full_policy_map_equal"])) for row in group
                ),
                "root_action_disagreement_rate": mean(
                    int(row["root_action_disagree"]) for row in group
                ),
            }
            for metric in METRIC_NAMES:
                field = f"delta_{metric}_regret_minus_cost"
                record[f"mean_delta_{metric}"] = mean(
                    float(row[field]) for row in group
                )
                record[f"disagreement_mean_delta_{metric}"] = conditional_mean(
                    group, field, disagreed
                )
            output.append(record)
    return output


def summarize_behavior(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for alpha in ALPHAS:
        for p_open in P_OPEN_VALUES:
            group = [
                row for row in rows
                if row["alpha"] == alpha and row["p_open"] == p_open
            ]
            disagreed = lambda row: int(row["behavior_disagree"]) == 1
            record = {
                "alpha": alpha,
                "p_open": p_open,
                "instances": len(group),
                "behavior_disagreement_rate": mean(
                    int(row["behavior_disagree"]) for row in group
                ),
                "root_action_disagreement_rate": mean(
                    int(row["root_action_disagree"]) for row in group
                ),
                "mean_disagreement_prior_mass": mean(
                    float(row["behavior_disagreement_prior_mass"]) for row in group
                ),
                "conditional_disagreement_prior_mass": conditional_mean(
                    group, "behavior_disagreement_prior_mass", disagreed
                ),
            }
            for metric in METRIC_NAMES:
                field = f"delta_{metric}_regret_minus_cost"
                record[f"disagreement_delta_{metric}"] = conditional_mean(
                    group, field, disagreed
                )
            output.append(record)
    return output


def summarize_by_alpha(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for alpha in ALPHAS:
        group = [row for row in rows if row["alpha"] == alpha]
        disagreed = [row for row in group if int(row["behavior_disagree"]) == 1]
        output.append({
            "alpha": alpha,
            "instances": len(group),
            "behavior_disagreement_rate": mean(
                int(row["behavior_disagree"]) for row in group
            ),
            "root_disagreement_rate": mean(
                int(row["root_action_disagree"]) for row in group
            ),
            "disagreement_mean_prior_mass": (
                mean(float(row["behavior_disagreement_prior_mass"]) for row in disagreed)
                if disagreed else ""
            ),
            "delta_expected_regret_on_disagreement": (
                mean(
                    float(row["delta_mean_regret_regret_minus_cost"])
                    for row in disagreed
                ) if disagreed else ""
            ),
            "delta_worst_cost_on_disagreement": (
                mean(
                    float(row["delta_worst_cost_regret_minus_cost"])
                    for row in disagreed
                ) if disagreed else ""
            ),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstructed deterministic driver for Robotica Table 10."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--jitter", type=float, default=0.05,
        help="Seed-specific half-width around each recorded p_open center.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows, behavior_rows = generate_raw(args.jitter)
    write_rows(output_dir / "stage3_3_p2_3_cost_vs_regret_raw.csv", policy_rows)
    write_rows(
        output_dir / "stage3_3_p2_3_cost_vs_regret_summary.csv",
        summarize_policy(policy_rows),
    )
    write_rows(
        output_dir / "stage3_3_p2_3_behavioral_disagreement_raw.csv",
        behavior_rows,
    )
    write_rows(
        output_dir / "stage3_3_p2_3_behavioral_disagreement_summary.csv",
        summarize_behavior(behavior_rows),
    )
    write_rows(
        output_dir / "stage3_3_p2_3_disagreement_by_alpha.csv",
        summarize_by_alpha(behavior_rows),
    )
    print("[PASS:RECOMPUTED] Stage 3.3 P2.3: 200 paired exact cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
