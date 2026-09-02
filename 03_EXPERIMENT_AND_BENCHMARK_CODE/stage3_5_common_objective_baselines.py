from __future__ import annotations

import argparse
import ast
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for import_dir in (CORE_DIR, EXPERIMENT_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from factored_belief import MixtureProductPrior
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage0_solver import AgentNode, DFA, EnvNode, PKWTS, World
from stage1_7_utils import (
    eventually_goal_dfa,
    independent_prior,
    random_multishortcut,
)
from stage3_3_baseline_solvers import (
    FactoredDynamicCVaRCostSolver,
    FactoredWorstCaseCostSolver,
    FactoredWorstCaseRegretSolver,
)
from stage3_3_intel_lab_benchmark import (
    firefighting_dfa,
    intel_lab_topological_pkwts,
    intel_two_mode_prior,
)
from stage3_5_horizon_sensitivity import format_number, write_csv
from stage3_5_prior_robustness import evaluate_fixed_policy


COMMON_PLANNING_ALPHA = 0.50
TERMINAL_EVALUATION_ALPHA = 0.95
HUB_HORIZON = 5
INTEL_HORIZON = 11
HUB_SEEDS = tuple(range(10))
VALUE_TOLERANCE = 1e-9
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reproduced_results"
OUTPUT_FILENAME = "stage3_5_common_objective_baselines.csv"
FROZEN_RAW_PATH = (
    PROJECT_ROOT
    / "06_DATA_AND_RESULTS"
    / "stage3_3_p0_baseline_comparison_raw.csv"
)
FROZEN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "06_DATA_AND_RESULTS"
    / "stage3_3_p0_baseline_comparison_summary.csv"
)


METHOD_ORDER = (
    "Worst absolute cost",
    "Minimax regret",
    "Expected cost / regret",
    "Dynamic CVaR cost a=0.5",
    "Proposed dynamic CVaR regret a=0.5",
)


OUTPUT_FIELDS = (
    "family",
    "instance_id",
    "seed",
    "method",
    "original_objective",
    "original_planning_alpha",
    "common_objective",
    "common_objective_planning_alpha",
    "terminal_evaluation_statistic",
    "terminal_evaluation_alpha",
    "horizon",
    "worlds",
    "root_action",
    "original_objective_value",
    "expected_regret",
    "static_cvar95_regret",
    "worst_regret",
    "common_proposed_nested_dynamic_regret",
    "common_objective_gap_to_proposed",
    "common_objective_optimal",
    "hard_satisfaction_flag",
    "exact_world_enumeration",
    "frozen_raw_row_match",
)


@dataclass(frozen=True)
class Benchmark:
    family: str
    instance_id: str
    seed: int
    transition_system: PKWTS
    dfa: DFA
    factored_prior: MixtureProductPrior
    explicit_prior: Mapping[World, float]
    horizon: int


@dataclass(frozen=True)
class SolvedPolicy:
    method: str
    original_objective: str
    original_planning_alpha: float | None
    policy: Mapping[Tuple[AgentNode, int], EnvNode]
    root_action: str
    original_objective_value: float


def benchmark_instances() -> Iterable[Benchmark]:
    dfa = eventually_goal_dfa()
    for seed in HUB_SEEDS:
        transition_system = random_multishortcut(8, seed)
        factored_prior = independent_prior(
            transition_system,
            p_open=0.30,
            seed=seed,
            jitter=0.08,
        )
        yield Benchmark(
            family="random_hub_m8",
            instance_id=f"hub_seed_{seed:02d}",
            seed=seed,
            transition_system=transition_system,
            dfa=dfa,
            factored_prior=factored_prior,
            explicit_prior=factored_prior.explicit_prior_for_validation(),
            horizon=HUB_HORIZON,
        )

    transition_system, _ = intel_lab_topological_pkwts()
    factored_prior = intel_two_mode_prior(transition_system)
    yield Benchmark(
        family="intel_map_scLTL",
        instance_id="intel_nominal",
        seed=0,
        transition_system=transition_system,
        dfa=firefighting_dfa(),
        factored_prior=factored_prior,
        explicit_prior=factored_prior.explicit_prior_for_validation(),
        horizon=INTEL_HORIZON,
    )


def solved_policy(
    method: str,
    benchmark: Benchmark,
) -> SolvedPolicy:
    transition_system = benchmark.transition_system
    dfa = benchmark.dfa
    prior = benchmark.factored_prior
    horizon = benchmark.horizon

    if method == "Worst absolute cost":
        solver = FactoredWorstCaseCostSolver(
            transition_system,
            dfa,
            prior,
            COMMON_PLANNING_ALPHA,
            horizon,
        )
        result = solver.solve_cost()
        objective = "worst_absolute_mission_cost"
        original_alpha = None
        objective_value = result.objective
    elif method == "Minimax regret":
        solver = FactoredWorstCaseRegretSolver(
            transition_system,
            dfa,
            prior,
            COMMON_PLANNING_ALPHA,
            horizon,
        )
        result = solver.solve_regret()
        objective = "worst_hindsight_regret"
        original_alpha = None
        objective_value = result.objective
    elif method == "Expected cost / regret":
        solver = FactoredDynamicCVaRCostSolver(
            transition_system,
            dfa,
            prior,
            0.0,
            horizon,
        )
        result = solver.solve_cost()
        objective = "expected_absolute_cost_equiv_expected_regret"
        original_alpha = 0.0
        objective_value = result.objective
    elif method == "Dynamic CVaR cost a=0.5":
        solver = FactoredDynamicCVaRCostSolver(
            transition_system,
            dfa,
            prior,
            COMMON_PLANNING_ALPHA,
            horizon,
        )
        result = solver.solve_cost()
        objective = "nested_dynamic_absolute_cost"
        original_alpha = COMMON_PLANNING_ALPHA
        objective_value = result.objective
    elif method == "Proposed dynamic CVaR regret a=0.5":
        solver = FactoredLazyHorizonDynamicCVaRSolver(
            transition_system,
            dfa,
            prior,
            COMMON_PLANNING_ALPHA,
            horizon,
        )
        result = solver.solve()
        objective = "nested_dynamic_hindsight_regret"
        original_alpha = COMMON_PLANNING_ALPHA
        objective_value = result.dynamic_regret_value
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    policy = result.policy
    root_action = policy[(solver.start, horizon)].target
    return SolvedPolicy(
        method=method,
        original_objective=objective,
        original_planning_alpha=original_alpha,
        policy=policy,
        root_action=root_action,
        original_objective_value=float(objective_value),
    )


def load_frozen_raw() -> Dict[Tuple[str, int, str], Mapping[str, str]]:
    with FROZEN_RAW_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["family"], int(row["seed"]), row["method"]): row
        for row in rows
    }


def assert_close(label: str, actual: float, expected: float) -> None:
    if not isclose(actual, expected, rel_tol=0.0, abs_tol=VALUE_TOLERANCE):
        raise RuntimeError(
            f"Frozen baseline mismatch for {label}: actual={actual!r}, "
            f"expected={expected!r}."
        )


def validate_frozen_raw_row(
    benchmark: Benchmark,
    solution: SolvedPolicy,
    expected_regret: float,
    cvar95_regret: float,
    worst_regret: float,
    all_worlds_satisfied: bool,
    frozen_raw: Mapping[Tuple[str, int, str], Mapping[str, str]],
) -> None:
    key = (benchmark.family, benchmark.seed, solution.method)
    if key not in frozen_raw:
        raise RuntimeError(f"Missing frozen raw baseline row: {key!r}")
    frozen = frozen_raw[key]
    if solution.root_action != frozen["root_action"]:
        raise RuntimeError(
            f"Frozen root-action mismatch for {key!r}: "
            f"{solution.root_action!r} != {frozen['root_action']!r}."
        )
    assert_close(
        f"{key!r} original objective",
        solution.original_objective_value,
        float(frozen["optimization_objective"]),
    )
    assert_close(
        f"{key!r} expected regret",
        expected_regret,
        float(frozen["mean_regret"]),
    )
    assert_close(
        f"{key!r} static CVaR.95 regret",
        cvar95_regret,
        float(frozen["cvar95_regret"]),
    )
    assert_close(
        f"{key!r} worst regret",
        worst_regret,
        float(frozen["worst_regret"]),
    )
    if int(all_worlds_satisfied) != int(frozen["satisfaction_all_worlds"]):
        raise RuntimeError(f"Frozen hard-satisfaction mismatch for {key!r}.")
    if len(benchmark.explicit_prior) != int(frozen["worlds"]):
        raise RuntimeError(f"Frozen world-count mismatch for {key!r}.")


def validate_frozen_summary(rows: Sequence[Mapping[str, object]]) -> None:
    with FROZEN_SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        frozen_rows = list(csv.DictReader(handle))
    grouped: Dict[Tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["method"]))].append(row)

    for frozen in frozen_rows:
        key = (frozen["family"], frozen["method"])
        if key not in grouped:
            raise RuntimeError(f"Missing common-objective rows for summary key {key!r}.")
        current = grouped[key]
        if len(current) != int(frozen["n_seeds"]):
            raise RuntimeError(f"Frozen seed-count mismatch for {key!r}.")
        for current_field, frozen_field in (
            ("expected_regret", "mean_regret"),
            ("static_cvar95_regret", "cvar95_regret"),
            ("worst_regret", "worst_regret"),
        ):
            assert_close(
                f"{key!r} summary {current_field}",
                mean(float(row[current_field]) for row in current),
                float(frozen[frozen_field]),
            )
        current_roots = Counter(str(row["root_action"]) for row in current)
        frozen_roots = Counter(ast.literal_eval(frozen["root_action_counts"]))
        if current_roots != frozen_roots:
            raise RuntimeError(f"Frozen root-count mismatch for {key!r}.")
        current_satisfaction = all(
            bool(row["hard_satisfaction_flag"]) for row in current
        )
        frozen_satisfaction = frozen["all_worlds_satisfied"].strip().lower() == "true"
        if current_satisfaction != frozen_satisfaction:
            raise RuntimeError(f"Frozen satisfaction-summary mismatch for {key!r}.")


def compute_rows() -> list[Dict[str, object]]:
    frozen_raw = load_frozen_raw()
    rows: list[Dict[str, object]] = []
    benchmarks = tuple(benchmark_instances())
    total = len(benchmarks) * len(METHOD_ORDER)
    completed = 0

    for benchmark in benchmarks:
        evaluator = LazyHorizonDynamicCVaRSolver(
            benchmark.transition_system,
            benchmark.dfa,
            benchmark.explicit_prior,
            COMMON_PLANNING_ALPHA,
            benchmark.horizon,
        )
        instance_rows: list[Dict[str, object]] = []
        for method in METHOD_ORDER:
            solution = solved_policy(method, benchmark)
            evaluation = evaluate_fixed_policy(evaluator, solution.policy)
            if evaluation.status != "complete":
                raise RuntimeError(
                    f"Incomplete fixed-policy evaluation for "
                    f"{benchmark.instance_id}/{method}: {evaluation.failure_reasons}"
                )
            if any(
                value is None
                for value in (
                    evaluation.dynamic_objective,
                    evaluation.mean_regret,
                    evaluation.cvar95_regret,
                    evaluation.worst_regret,
                )
            ):
                raise RuntimeError(
                    f"Missing exact metric for {benchmark.instance_id}/{method}."
                )

            common_value = float(evaluation.dynamic_objective)
            expected_regret = float(evaluation.mean_regret)
            cvar95_regret = float(evaluation.cvar95_regret)
            worst_regret = float(evaluation.worst_regret)
            validate_frozen_raw_row(
                benchmark=benchmark,
                solution=solution,
                expected_regret=expected_regret,
                cvar95_regret=cvar95_regret,
                worst_regret=worst_regret,
                all_worlds_satisfied=evaluation.all_worlds_satisfied,
                frozen_raw=frozen_raw,
            )
            instance_rows.append(
                {
                    "family": benchmark.family,
                    "instance_id": benchmark.instance_id,
                    "seed": benchmark.seed,
                    "method": solution.method,
                    "original_objective": solution.original_objective,
                    "original_planning_alpha": solution.original_planning_alpha,
                    "common_objective": "proposed_nested_dynamic_hindsight_regret",
                    "common_objective_planning_alpha": COMMON_PLANNING_ALPHA,
                    "terminal_evaluation_statistic": "static_terminal_cvar_regret",
                    "terminal_evaluation_alpha": TERMINAL_EVALUATION_ALPHA,
                    "horizon": benchmark.horizon,
                    "worlds": len(benchmark.explicit_prior),
                    "root_action": solution.root_action,
                    "original_objective_value": solution.original_objective_value,
                    "expected_regret": expected_regret,
                    "static_cvar95_regret": cvar95_regret,
                    "worst_regret": worst_regret,
                    "common_proposed_nested_dynamic_regret": common_value,
                    "hard_satisfaction_flag": evaluation.all_worlds_satisfied,
                    "exact_world_enumeration": True,
                    "frozen_raw_row_match": True,
                }
            )
            completed += 1
            print(
                f"[PROGRESS] {completed}/{total} common-objective policies",
                flush=True,
            )

        proposed_value = next(
            float(row["common_proposed_nested_dynamic_regret"])
            for row in instance_rows
            if row["method"] == "Proposed dynamic CVaR regret a=0.5"
        )
        for row in instance_rows:
            gap = float(row["common_proposed_nested_dynamic_regret"]) - proposed_value
            if gap < -VALUE_TOLERANCE:
                raise RuntimeError(
                    f"A baseline beats the exact common-objective optimum for "
                    f"{benchmark.instance_id}/{row['method']}: gap={gap!r}."
                )
            row["common_objective_gap_to_proposed"] = (
                0.0 if abs(gap) <= VALUE_TOLERANCE else gap
            )
            row["common_objective_optimal"] = abs(gap) <= VALUE_TOLERANCE
        rows.extend(instance_rows)

    validate_frozen_summary(rows)
    return rows


def csv_rows(rows: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    numeric_fields = {
        "original_objective_value",
        "expected_regret",
        "static_cvar95_regret",
        "worst_regret",
        "common_proposed_nested_dynamic_regret",
        "common_objective_gap_to_proposed",
    }
    boolean_fields = {
        "common_objective_optimal",
        "hard_satisfaction_flag",
        "exact_world_enumeration",
        "frozen_raw_row_match",
    }
    formatted = []
    for row in rows:
        output = dict(row)
        original_alpha = output["original_planning_alpha"]
        output["original_planning_alpha"] = (
            "" if original_alpha is None else format_number(float(original_alpha))
        )
        output["common_objective_planning_alpha"] = format_number(
            float(output["common_objective_planning_alpha"])
        )
        output["terminal_evaluation_alpha"] = format_number(
            float(output["terminal_evaluation_alpha"])
        )
        for field in numeric_fields:
            output[field] = format_number(float(output[field]))
        for field in boolean_fields:
            output[field] = int(bool(output[field]))
        formatted.append(output)
    return formatted


def print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: Dict[Tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["method"]))].append(
            float(row["common_proposed_nested_dynamic_regret"])
        )
    print("[SUMMARY] Common proposed nested objective at planning alpha=0.5")
    for method in METHOD_ORDER:
        hub_value = mean(grouped[("random_hub_m8", method)])
        intel_value = mean(grouped[("intel_map_scLTL", method)])
        print(
            f"[SUMMARY] {method}: hub_mean={hub_value:.12f}, "
            f"intel={intel_value:.12f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate all frozen H-proper baseline policies under the "
            "proposed nested dynamic-regret objective at planning alpha=0.5."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = compute_rows()
    output_path = args.output_dir.resolve() / OUTPUT_FILENAME
    write_csv(output_path, OUTPUT_FIELDS, csv_rows(rows))
    print(f"[PASS] Wrote {len(rows)} exact rows to {output_path}")
    print("[PASS] All 55 rows match the frozen raw baseline metrics and roots.")
    print("[PASS] All 10 frozen family/method summaries match.")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
