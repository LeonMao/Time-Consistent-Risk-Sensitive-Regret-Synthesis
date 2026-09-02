from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
from statistics import mean, stdev

from stage3_3_reconstructed_common import PROJECT_ROOT, exact_policy_metrics
from stage1_7_utils import eventually_goal_dfa, independent_prior, random_multishortcut
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver


ALPHAS = (0.0, 0.25, 0.5, 0.75, 0.9)
SEEDS = tuple(range(10))
RAW_FIELDS = (
    "seed", "m", "worlds", "H", "planning_alpha", "root_action",
    "dynamic_objective", "exact_mean_cost", "exact_mean_regret",
    "exact_var95_regret", "exact_cvar95_regret", "exact_worst_regret",
    "exact_satisfaction_all_worlds",
)
SUMMARY_FIELDS = (
    "planning_alpha", "dynamic_objective_mean", "dynamic_objective_sd",
    "exact_mean_cost_mean", "exact_mean_cost_sd", "exact_mean_regret_mean",
    "exact_mean_regret_sd", "exact_cvar95_regret_mean",
    "exact_cvar95_regret_sd", "exact_worst_regret_mean",
    "exact_worst_regret_sd", "root_action_counts", "all_worlds_satisfied",
)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_family(m: int) -> tuple[list[dict], list[dict]]:
    raw: list[dict] = []
    dfa = eventually_goal_dfa()
    for seed in SEEDS:
        transition_system = random_multishortcut(m, seed)
        prior = independent_prior(
            transition_system, 0.30, seed=seed, jitter=0.08
        )
        for alpha in ALPHAS:
            solver = FactoredLazyHorizonDynamicCVaRSolver(
                transition_system, dfa, prior, alpha=alpha, horizon=5
            )
            result = solver.solve()
            metrics = exact_policy_metrics(solver, result.policy)
            raw.append({
                "seed": seed,
                "m": m,
                "worlds": metrics["worlds"],
                "H": 5,
                "planning_alpha": alpha,
                "root_action": result.policy[(solver.start, 5)].target,
                "dynamic_objective": result.dynamic_regret_value,
                "exact_mean_cost": metrics["mean_cost"],
                "exact_mean_regret": metrics["mean_regret"],
                "exact_var95_regret": metrics["var95_regret"],
                "exact_cvar95_regret": metrics["cvar95_regret"],
                "exact_worst_regret": metrics["worst_regret"],
                "exact_satisfaction_all_worlds": metrics[
                    "satisfaction_all_worlds"
                ],
            })

    summary: list[dict] = []
    for alpha in ALPHAS:
        selected = [row for row in raw if row["planning_alpha"] == alpha]
        dynamic = [float(row["dynamic_objective"]) for row in selected]
        costs = [float(row["exact_mean_cost"]) for row in selected]
        mean_regrets = [float(row["exact_mean_regret"]) for row in selected]
        cvar_regrets = [float(row["exact_cvar95_regret"]) for row in selected]
        worst_regrets = [float(row["exact_worst_regret"]) for row in selected]
        counts = Counter(str(row["root_action"]) for row in selected)
        summary.append({
            "planning_alpha": alpha,
            "dynamic_objective_mean": mean(dynamic),
            "dynamic_objective_sd": stdev(dynamic),
            "exact_mean_cost_mean": mean(costs),
            "exact_mean_cost_sd": stdev(costs),
            "exact_mean_regret_mean": mean(mean_regrets),
            "exact_mean_regret_sd": stdev(mean_regrets),
            "exact_cvar95_regret_mean": mean(cvar_regrets),
            "exact_cvar95_regret_sd": stdev(cvar_regrets),
            "exact_worst_regret_mean": mean(worst_regrets),
            "exact_worst_regret_sd": stdev(worst_regrets),
            "root_action_counts": str(dict(counts)),
            "all_worlds_satisfied": all(
                int(row["exact_satisfaction_all_worlds"]) == 1 for row in selected
            ),
        })
    return raw, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstructed deterministic driver for Robotica Fig. 5."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, nargs="+", choices=(8, 10), default=(8, 10))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for m in args.m:
        raw, summary = run_family(m)
        prefix = f"stage3_3_p0_exact_alpha_m{m}"
        write_csv(output_dir / f"{prefix}.csv", RAW_FIELDS, raw)
        summary_fields = SUMMARY_FIELDS
        if m == 10:
            omitted = {"exact_mean_cost_mean", "exact_mean_cost_sd"}
            summary_fields = tuple(
                field for field in SUMMARY_FIELDS if field not in omitted
            )
            summary = [
                {field: row[field] for field in summary_fields}
                for row in summary
            ]
        write_csv(output_dir / f"{prefix}_summary.csv", summary_fields, summary)
        print(f"[PASS:RECOMPUTED] m={m}: {len(raw)} exact cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
