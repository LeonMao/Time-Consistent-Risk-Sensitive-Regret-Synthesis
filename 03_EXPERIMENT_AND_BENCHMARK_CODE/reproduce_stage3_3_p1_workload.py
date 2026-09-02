from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
for import_dir in (EXPERIMENT_DIR, CORE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage1_7_utils import (
    eventually_goal_dfa as hub_goal_dfa,
    independent_prior as hub_prior,
    random_multishortcut,
)
from stage3_2_benchmark_families import (
    distributed_layered_pkwts,
    eventually_goal_dfa as layered_goal_dfa,
    two_mode_correlated_prior,
)


@dataclass(frozen=True)
class Config:
    family: str
    m: int
    horizon: int
    layers: int | None = None
    width: int | None = None


CONFIGS = (
    Config("hub", 8, 3),
    Config("hub", 10, 3),
    Config("layered", 8, 6, 5, 2),
    Config("layered", 10, 7, 6, 2),
)
SCALE_CONFIG = Config("layered", 12, 8, 7, 2)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build(config: Config, seed: int):
    if config.family == "hub":
        transition_system = random_multishortcut(config.m, seed)
        dfa = hub_goal_dfa()
        prior = hub_prior(
            transition_system, 0.30, seed=seed, jitter=0.05
        )
        return transition_system, dfa, prior
    if config.layers is None or config.width is None:
        raise RuntimeError("Layered configuration is incomplete.")
    transition_system = distributed_layered_pkwts(
        config.layers, config.width, seed
    )
    return (
        transition_system,
        layered_goal_dfa(),
        two_mode_correlated_prior(transition_system, seed),
    )


def policy_map(policy) -> dict:
    return {key: str(env.target) for key, env in policy.items()}


def probability_entry_count(prior) -> int:
    mode_entries = len(prior._mode_cache)
    marginal_entries = sum(
        len(value) for value in prior._marginal_cache.values()
    )
    return mode_entries + marginal_entries


def run_pair(config: Config, seed: int):
    transition_system, dfa, prior = build(config, seed)
    explicit_prior = prior.explicit_prior_for_validation()
    factored_solver = FactoredLazyHorizonDynamicCVaRSolver(
        transition_system, dfa, prior, alpha=0.5, horizon=config.horizon,
        use_lower_bound_pruning=False,
    )
    factored = factored_solver.solve()
    explicit_solver = LazyHorizonDynamicCVaRSolver(
        transition_system, dfa, explicit_prior, alpha=0.5,
        horizon=config.horizon, use_lower_bound_pruning=False
    )
    explicit = explicit_solver.solve()

    factored_map = policy_map(factored.policy)
    explicit_map = policy_map(explicit.policy)
    common = set(factored_map) & set(explicit_map)
    different = sum(factored_map[key] != explicit_map[key] for key in common)
    value_error = abs(
        float(factored.dynamic_regret_value)
        - float(explicit.dynamic_regret_value)
    )

    operation_rows = [
        {
            "family": config.family,
            "m": config.m,
            "seed": seed,
            "worlds": len(explicit_prior),
            "method": "factored",
            "generated_agent_states": factored.generated_agent_states,
            "generated_observation_branches": factored.generated_observation_branches,
            "value_state_budgets": factored.value_state_budgets,
            "pruned_actions": factored.pruned_actions,
            "oracle_shortest_path_calls": factored.oracle_shortest_path_calls,
            "oracle_symbolic_nodes": (
                factored.symbolic_oracle_nodes // len(prior.components)
            ),
        },
        {
            "family": config.family,
            "m": config.m,
            "seed": seed,
            "worlds": len(explicit_prior),
            "method": "explicit",
            "generated_agent_states": explicit.generated_agent_states,
            "generated_observation_branches": explicit.generated_action_branches,
            "value_state_budgets": explicit.value_expanded_state_budgets,
            "pruned_actions": explicit.pruned_actions_by_bound,
            "oracle_shortest_path_calls": len(explicit_prior),
            "oracle_symbolic_nodes": len(explicit_prior),
        },
    ]
    explicit_entries = sum(
        len(value) for value in explicit_solver._posterior_cache.values()
    )
    factored_entries = probability_entry_count(prior)
    workload_row = {
        "family": config.family,
        "m": config.m,
        "seed": seed,
        "worlds": len(explicit_prior),
        "explicit_posterior_mass_entries": explicit_entries,
        "explicit_posterior_states": len(explicit_solver._posterior_cache),
        "factored_probability_entries": factored_entries,
        "factored_mode_posterior_states": len(prior._mode_cache),
        "factored_marginal_queries": len(prior._marginal_cache),
        "posterior_entry_reduction_x": explicit_entries / factored_entries,
    }
    exactness_row = {
        "family": config.family,
        "m": config.m,
        "seed": seed,
        "H": config.horizon,
        "factored_policy_entries": len(factored_map),
        "explicit_policy_entries": len(explicit_map),
        "policy_key_sets_equal": int(set(factored_map) == set(explicit_map)),
        "different_action_entries": different,
        "full_computed_policy_map_equal": int(factored_map == explicit_map),
        "value_abs_error": value_error,
    }
    return operation_rows, workload_row, exactness_row


def summarize_operations(rows: list[dict]) -> list[dict]:
    output = []
    for config in CONFIGS:
        for method in ("factored", "explicit"):
            group = [
                row for row in rows
                if row["family"] == config.family
                and row["m"] == config.m and row["method"] == method
            ]
            output.append({
                "family": config.family,
                "m": config.m,
                "method": method,
                "worlds": 2 ** config.m,
                "agent_states_median": median(
                    row["generated_agent_states"] for row in group
                ),
                "observation_branches_median": median(
                    row["generated_observation_branches"] for row in group
                ),
                "value_state_budgets_median": median(
                    row["value_state_budgets"] for row in group
                ),
                "pruned_actions_median": median(
                    row["pruned_actions"] for row in group
                ),
                "oracle_calls_median": median(
                    row["oracle_shortest_path_calls"] for row in group
                ),
                "oracle_nodes_median": median(
                    row["oracle_symbolic_nodes"] for row in group
                ),
            })
    return output


def summarize_workload(rows: list[dict]) -> list[dict]:
    output = []
    for config in CONFIGS:
        group = [
            row for row in rows
            if row["family"] == config.family and row["m"] == config.m
        ]
        output.append({
            "family": config.family,
            "m": config.m,
            "worlds": 2 ** config.m,
            "explicit_posterior_mass_entries_median": median(
                row["explicit_posterior_mass_entries"] for row in group
            ),
            "factored_probability_entries_median": median(
                row["factored_probability_entries"] for row in group
            ),
            "posterior_entry_reduction_x_median": median(
                row["posterior_entry_reduction_x"] for row in group
            ),
            "explicit_posterior_states_median": median(
                row["explicit_posterior_states"] for row in group
            ),
            "factored_mode_states_median": median(
                row["factored_mode_posterior_states"] for row in group
            ),
            "factored_marginal_queries_median": median(
                row["factored_marginal_queries"] for row in group
            ),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstructed structural-workload driver for Robotica Table 11."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    operation_rows: list[dict] = []
    workload_rows: list[dict] = []
    exactness_rows: list[dict] = []
    for config in CONFIGS:
        for seed in range(5):
            operations, workload, exactness = run_pair(config, seed)
            operation_rows.extend(operations)
            workload_rows.append(workload)
            exactness_rows.append(exactness)
    _, _, scale_exactness = run_pair(SCALE_CONFIG, 0)
    exactness_rows.append(scale_exactness)

    write_rows(
        output_dir / "stage3_3_p1_operation_counts_raw.csv", operation_rows
    )
    write_rows(
        output_dir / "stage3_3_p1_operation_counts_summary.csv",
        summarize_operations(operation_rows),
    )
    write_rows(
        output_dir / "stage3_3_p1_posterior_workload_raw.csv", workload_rows
    )
    write_rows(
        output_dir / "stage3_3_p1_posterior_workload_summary.csv",
        summarize_workload(workload_rows),
    )
    write_rows(
        output_dir / "stage3_3_p0_full_policy_exactness.csv", exactness_rows
    )
    print(
        "[PASS:RECOMPUTED] Stage 3.3 P1 workload: "
        f"{len(operation_rows)} method rows; {len(exactness_rows)} policy-map pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
