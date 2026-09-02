from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from math import inf, isclose
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Iterable, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for import_dir in (EXPERIMENT_DIR, CORE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage0_solver import AgentNode, DFA, EnvNode, PKWTS, World
from stage1_3_risk import discrete_cvar
from stage1_7_utils import (
    eventually_goal_dfa as hub_goal_dfa,
    independent_prior as hub_independent_prior,
    ordered_stage_benchmark,
    random_multishortcut,
)
from stage3_2_benchmark_families import (
    distributed_layered_pkwts,
    eventually_goal_dfa as layered_goal_dfa,
    independent_prior as layered_independent_prior,
    unknown_count,
)
from stage3_3_intel_lab_benchmark import (
    firefighting_dfa,
    intel_lab_topological_pkwts,
    intel_two_mode_prior,
)


DEFAULT_ALPHAS = (0.0, 0.25, 0.50, 0.75, 0.90)
DEFAULT_MAX_HORIZON = 13
DEFAULT_REPETITIONS = 3
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reproduced_results"
INTEL_FILENAME = "stage3_5_intel_horizon_sensitivity.csv"
RANK_FILENAME = "stage3_5_minimal_robust_ranks.csv"
VALUE_TOLERANCE = 1e-10


RANK_FIELDS = (
    "family",
    "instance_id",
    "benchmark_role",
    "seed",
    "state_count",
    "unknown_variable_count",
    "supported_world_count",
    "dfa_state_count",
    "prior_name",
    "minimal_robust_rank",
    "rank_minus_one_feasible",
    "rank_feasible",
    "reference_horizon",
    "reference_horizon_relation",
    "rank_search_limit",
    "generated_agent_states",
    "generated_env_actions",
    "generated_observation_branches",
    "exact_supported_world_enumeration",
)


INTEL_FIELDS = (
    "instance_id",
    "alpha",
    "horizon",
    "minimal_robust_rank",
    "horizon_slack_above_minimal_rank",
    "world_count",
    "dfa_state_count",
    "dynamic_regret_objective",
    "root_action",
    "reachable_policy_state_count",
    "reachable_policy_map_sha256",
    "reachable_policy_map_json",
    "generated_agent_states",
    "generated_env_actions",
    "generated_action_branches",
    "value_expanded_state_budgets",
    "pruned_actions_by_bound",
    "exact_mean_regret",
    "exact_cvar95_regret",
    "exact_worst_regret",
    "all_worlds_satisfied",
    "warmup_runs",
    "timed_repetitions",
    "initialization_median_s",
    "solve_median_s",
    "total_median_s",
    "total_min_s",
    "total_max_s",
    "runtime_scope",
    "comparison_max_horizon",
    "value_equal_to_max_horizon",
    "reachable_policy_equal_to_max_horizon",
    "joint_value_policy_equal_to_max_horizon",
    "first_value_stable_h_through_max",
    "first_policy_stable_h_through_max",
    "first_joint_stable_h_through_max",
    "joint_stabilization_observed_through_max",
    "exact_world_enumeration",
)


@dataclass(frozen=True)
class RankBenchmark:
    family: str
    instance_id: str
    benchmark_role: str
    seed: int | None
    transition_system: PKWTS
    dfa: DFA
    prior: Mapping[World, float]
    prior_name: str
    reference_horizon: int | None


class ExactSupportedWorldRobustRank:
    def __init__(
        self,
        transition_system: PKWTS,
        dfa: DFA,
        prior: Mapping[World, float],
    ):
        self.transition_system = transition_system
        self.dfa = dfa
        self.worlds = tuple(
            sorted(world for world, probability in prior.items() if probability > 0)
        )
        if not self.worlds:
            raise ValueError("The positive-probability support is empty.")
        initial_q = dfa.step(dfa.q_init, transition_system.labels[transition_system.x0])
        self.start = AgentNode(
            transition_system.x0,
            initial_q,
            frozenset(),
        )
        self._compatible_cache: Dict[frozenset, Tuple[World, ...]] = {}
        self._branch_cache: Dict[EnvNode, Tuple[AgentNode, ...]] = {}
        self._feasible_cache: Dict[Tuple[AgentNode, int], bool] = {}
        self.generated_agent_states = {self.start}
        self.generated_env_actions: set[EnvNode] = set()
        self.generated_observation_branches = 0

    def compatible_worlds(self, knowledge: frozenset) -> Tuple[World, ...]:
        if knowledge in self._compatible_cache:
            return self._compatible_cache[knowledge]
        evidence = dict(knowledge)
        compatible = tuple(
            world
            for world in self.worlds
            if all(
                self.transition_system.world_pattern_index(world, state_name)
                == pattern_index
                for state_name, pattern_index in evidence.items()
            )
        )
        if not compatible:
            raise RuntimeError("A generated knowledge state has empty support.")
        self._compatible_cache[knowledge] = compatible
        return compatible

    def current_successors(self, state: AgentNode) -> frozenset:
        patterns = self.transition_system.patterns[state.x]
        if len(patterns) == 1:
            return patterns[0]
        evidence = dict(state.K)
        if state.x not in evidence:
            raise RuntimeError(f"Unknown current state {state.x} has not been revealed.")
        return patterns[evidence[state.x]]

    def actions(self, state: AgentNode) -> Tuple[EnvNode, ...]:
        if state.q in self.dfa.accepting:
            return tuple()
        actions = tuple(
            EnvNode(state.x, state.q, state.K, target)
            for target in sorted(self.current_successors(state))
        )
        self.generated_env_actions.update(actions)
        return actions

    def child_for_world(self, action: EnvNode, world: World) -> AgentNode:
        pattern_index = self.transition_system.world_pattern_index(
            world,
            action.target,
        )
        knowledge = action.K
        if len(self.transition_system.patterns[action.target]) > 1:
            knowledge = frozenset(set(knowledge) | {(action.target, pattern_index)})
        child_q = self.dfa.step(
            action.q,
            self.transition_system.labels[action.target],
        )
        return AgentNode(action.target, child_q, knowledge)

    def branches(self, action: EnvNode) -> Tuple[AgentNode, ...]:
        if action in self._branch_cache:
            return self._branch_cache[action]
        children = {
            self.child_for_world(action, world)
            for world in self.compatible_worlds(action.K)
        }
        ordered = tuple(sorted(children, key=canonical_agent_state))
        if not ordered:
            raise RuntimeError("A generated action has no supported observation branch.")
        self._branch_cache[action] = ordered
        self.generated_agent_states.update(ordered)
        self.generated_observation_branches += len(ordered)
        return ordered

    def can_accept_within(self, state: AgentNode, depth: int) -> bool:
        key = (state, depth)
        if key in self._feasible_cache:
            return self._feasible_cache[key]
        if state.q in self.dfa.accepting:
            self._feasible_cache[key] = True
            return True
        if depth <= 0:
            self._feasible_cache[key] = False
            return False
        feasible = any(
            all(
                self.can_accept_within(child, depth - 1)
                for child in self.branches(action)
            )
            for action in self.actions(state)
        )
        self._feasible_cache[key] = feasible
        return feasible

    def minimal_rank(self, search_limit: int) -> int:
        for depth in range(search_limit + 1):
            if self.can_accept_within(self.start, depth):
                return depth
        return inf


def canonical_knowledge(knowledge: frozenset) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((str(name), int(pattern)) for name, pattern in knowledge))


def canonical_agent_state(state: AgentNode) -> Tuple[str, str, Tuple[Tuple[str, int], ...]]:
    return str(state.x), str(state.q), canonical_knowledge(state.K)


def format_number(value: float) -> str:
    return f"{value:.12g}"


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reference_horizon_relation(
    minimal_rank: int,
    reference_horizon: int | None,
) -> str:
    if reference_horizon is None:
        return "not_recorded_for_this_compact_benchmark"
    if reference_horizon == minimal_rank:
        return "minimal"
    if reference_horizon > minimal_rank:
        return f"conservative_by_{reference_horizon - minimal_rank}"
    return f"infeasible_below_minimum_by_{minimal_rank - reference_horizon}"


def build_rank_benchmarks() -> Iterable[RankBenchmark]:
    for unknowns, reference_horizon in ((8, 5), (10, 5)):
        for seed in range(10):
            transition_system = random_multishortcut(unknowns, seed)
            prior = hub_independent_prior(
                transition_system,
                p_open=0.5,
                seed=seed,
                jitter=0.0,
            ).explicit_prior_for_validation()
            yield RankBenchmark(
                family="random_multishortcut",
                instance_id=f"hub_m{unknowns}_seed{seed:03d}",
                benchmark_role=f"exact_risk_sweep_m{unknowns}",
                seed=seed,
                transition_system=transition_system,
                dfa=hub_goal_dfa(),
                prior=prior,
                prior_name="independent_full_support_p0.50",
                reference_horizon=reference_horizon,
            )

    layered_configs = (
        (5, 2, range(5), "controlled_layered_m8"),
        (6, 2, range(5), "controlled_layered_m10"),
        (7, 2, range(1), "controlled_layered_m12_stress"),
    )
    for layers, width, seeds, role in layered_configs:
        for seed in seeds:
            transition_system = distributed_layered_pkwts(layers, width, seed)
            prior = layered_independent_prior(
                transition_system,
                p_open=0.30,
                seed=seed,
                jitter=0.05,
            ).explicit_prior_for_validation()
            yield RankBenchmark(
                family="distributed_layered",
                instance_id=f"layered_L{layers}_W{width}_seed{seed:03d}",
                benchmark_role=role,
                seed=seed,
                transition_system=transition_system,
                dfa=layered_goal_dfa(),
                prior=prior,
                prior_name="independent_full_support_p0.30_jitter0.05",
                reference_horizon=None,
            )

    for ordered_goals in range(1, 5):
        for seed in range(3):
            transition_system, dfa = ordered_stage_benchmark(
                ordered_goals,
                shortcuts_per_stage=2,
                seed=seed,
            )
            prior = hub_independent_prior(
                transition_system,
                p_open=0.30,
                seed=seed,
                jitter=0.05,
            ).explicit_prior_for_validation()
            yield RankBenchmark(
                family="ordered_stage",
                instance_id=(
                    f"ordered_k{ordered_goals}_s2_seed{seed:03d}"
                ),
                benchmark_role="fixed_graph_dfa_scan",
                seed=seed,
                transition_system=transition_system,
                dfa=dfa,
                prior=prior,
                prior_name="independent_full_support_p0.30_jitter0.05",
                reference_horizon=4 * ordered_goals,
            )

    intel_system, _ = intel_lab_topological_pkwts()
    yield RankBenchmark(
        family="intel_lab_topological",
        instance_id="intel_lab_full_support",
        benchmark_role="map_derived_scltl",
        seed=None,
        transition_system=intel_system,
        dfa=firefighting_dfa(),
        prior=intel_two_mode_prior(intel_system).explicit_prior_for_validation(),
        prior_name="two_mode_full_support",
        reference_horizon=11,
    )


def compute_rank_rows() -> list[dict[str, object]]:
    rows = []
    for benchmark in build_rank_benchmarks():
        search_limit = max(
            4,
            2
            * len(benchmark.transition_system.states)
            * len(benchmark.dfa.states),
        )
        rank_solver = ExactSupportedWorldRobustRank(
            benchmark.transition_system,
            benchmark.dfa,
            benchmark.prior,
        )
        minimal_rank = rank_solver.minimal_rank(search_limit)
        if minimal_rank == inf:
            raise RuntimeError(
                f"No robust completion rank found for {benchmark.instance_id}."
            )
        rank_feasible = rank_solver.can_accept_within(
            rank_solver.start,
            minimal_rank,
        )
        prior_rank_feasible = (
            False
            if minimal_rank == 0
            else rank_solver.can_accept_within(
                rank_solver.start,
                minimal_rank - 1,
            )
        )
        if not rank_feasible or prior_rank_feasible:
            raise RuntimeError(
                f"Minimal-rank certificate failed for {benchmark.instance_id}."
            )
        rows.append(
            {
                "family": benchmark.family,
                "instance_id": benchmark.instance_id,
                "benchmark_role": benchmark.benchmark_role,
                "seed": "" if benchmark.seed is None else benchmark.seed,
                "state_count": len(benchmark.transition_system.states),
                "unknown_variable_count": unknown_count(
                    benchmark.transition_system
                ),
                "supported_world_count": len(benchmark.prior),
                "dfa_state_count": len(benchmark.dfa.states),
                "prior_name": benchmark.prior_name,
                "minimal_robust_rank": minimal_rank,
                "rank_minus_one_feasible": int(prior_rank_feasible),
                "rank_feasible": int(rank_feasible),
                "reference_horizon": (
                    ""
                    if benchmark.reference_horizon is None
                    else benchmark.reference_horizon
                ),
                "reference_horizon_relation": reference_horizon_relation(
                    minimal_rank,
                    benchmark.reference_horizon,
                ),
                "rank_search_limit": search_limit,
                "generated_agent_states": len(
                    rank_solver.generated_agent_states
                ),
                "generated_env_actions": len(
                    rank_solver.generated_env_actions
                ),
                "generated_observation_branches": (
                    rank_solver.generated_observation_branches
                ),
                "exact_supported_world_enumeration": 1,
            }
        )
    return rows


def child_for_world(
    solver: LazyHorizonDynamicCVaRSolver,
    action: EnvNode,
    world: World,
) -> AgentNode:
    pattern_index = solver.T.world_pattern_index(world, action.target)
    knowledge = action.K
    if len(solver.T.patterns[action.target]) > 1:
        knowledge = frozenset(set(knowledge) | {(action.target, pattern_index)})
    child_q = solver.A.step(action.q, solver.T.labels[action.target])
    return AgentNode(action.target, child_q, knowledge)


def reachable_policy_map(
    solver: LazyHorizonDynamicCVaRSolver,
    policy: Mapping[Tuple[AgentNode, int], EnvNode],
) -> Tuple[str, str, int]:
    reached: Dict[
        Tuple[int, str, str, Tuple[Tuple[str, int], ...]],
        str,
    ] = {}
    for world in solver.worlds:
        state = solver.start
        remaining = solver.horizon
        depth = 0
        while state.q not in solver.A.accepting:
            if remaining <= 0:
                raise RuntimeError("A reachable policy path exceeded its horizon.")
            action = policy[(state, remaining)]
            key = (
                depth,
                str(state.x),
                str(state.q),
                canonical_knowledge(state.K),
            )
            prior_action = reached.get(key)
            if prior_action is not None and prior_action != action.target:
                raise RuntimeError("One reachable information state has two actions.")
            reached[key] = action.target
            state = child_for_world(solver, action, world)
            remaining -= 1
            depth += 1

    entries = [
        {
            "depth": depth,
            "x": state_x,
            "q": state_q,
            "knowledge": [list(item) for item in knowledge],
            "action": reached[(depth, state_x, state_q, knowledge)],
        }
        for depth, state_x, state_q, knowledge in sorted(reached)
    ]
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()
    return payload, digest, len(entries)


def solve_intel_once(
    transition_system: PKWTS,
    dfa: DFA,
    prior: Mapping[World, float],
    alpha: float,
    horizon: int,
) -> dict[str, object]:
    total_started = time.perf_counter()
    solver = LazyHorizonDynamicCVaRSolver(
        transition_system,
        dfa,
        prior,
        alpha,
        horizon,
    )
    solve_started = time.perf_counter()
    result = solver.solve()
    solved_at = time.perf_counter()
    policy_json, policy_hash, policy_state_count = reachable_policy_map(
        solver,
        result.policy,
    )
    probabilities = [solver.prior[world] for world in solver.worlds]
    regrets = [result.policy_world_regrets[world] for world in solver.worlds]
    mean_regret = sum(
        probability * regret
        for probability, regret in zip(probabilities, regrets)
    )
    all_satisfied = True
    for world in solver.worlds:
        _, terminal = solver.simulate_budget_policy_world(world)
        all_satisfied = (
            all_satisfied
            and terminal.q in solver.A.accepting
        )
    return {
        "dynamic_regret_objective": result.dynamic_regret_value,
        "root_action": result.policy[(solver.start, horizon)].target,
        "reachable_policy_state_count": policy_state_count,
        "reachable_policy_map_sha256": policy_hash,
        "reachable_policy_map_json": policy_json,
        "generated_agent_states": result.generated_agent_states,
        "generated_env_actions": result.generated_env_actions,
        "generated_action_branches": result.generated_action_branches,
        "value_expanded_state_budgets": result.value_expanded_state_budgets,
        "pruned_actions_by_bound": result.pruned_actions_by_bound,
        "exact_mean_regret": mean_regret,
        "exact_cvar95_regret": discrete_cvar(
            regrets,
            probabilities,
            0.95,
        ),
        "exact_worst_regret": max(regrets),
        "all_worlds_satisfied": int(all_satisfied),
        "initialization_s": solve_started - total_started,
        "solve_s": solved_at - solve_started,
        "total_s": solved_at - total_started,
    }


def deterministic_signature(run: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        format_number(float(run["dynamic_regret_objective"])),
        run["root_action"],
        run["reachable_policy_map_sha256"],
        run["generated_agent_states"],
        run["generated_env_actions"],
        run["generated_action_branches"],
        run["value_expanded_state_budgets"],
        run["pruned_actions_by_bound"],
        format_number(float(run["exact_mean_regret"])),
        format_number(float(run["exact_cvar95_regret"])),
        format_number(float(run["exact_worst_regret"])),
        run["all_worlds_satisfied"],
    )


def measured_intel_configuration(
    transition_system: PKWTS,
    dfa: DFA,
    prior: Mapping[World, float],
    alpha: float,
    horizon: int,
    repetitions: int,
) -> dict[str, object]:
    solve_intel_once(
        transition_system,
        dfa,
        prior,
        alpha,
        horizon,
    )
    runs = [
        solve_intel_once(
            transition_system,
            dfa,
            prior,
            alpha,
            horizon,
        )
        for _ in range(repetitions)
    ]
    signatures = {deterministic_signature(run) for run in runs}
    if len(signatures) != 1:
        raise RuntimeError(
            f"Non-deterministic Intel result at alpha={alpha}, H={horizon}."
        )
    representative = dict(runs[0])
    initialization_times = [float(run["initialization_s"]) for run in runs]
    solve_times = [float(run["solve_s"]) for run in runs]
    total_times = [float(run["total_s"]) for run in runs]
    representative.update(
        {
            "warmup_runs": 1,
            "timed_repetitions": repetitions,
            "initialization_median_s": statistics.median(
                initialization_times
            ),
            "solve_median_s": statistics.median(solve_times),
            "total_median_s": statistics.median(total_times),
            "total_min_s": min(total_times),
            "total_max_s": max(total_times),
        }
    )
    for internal_field in ("initialization_s", "solve_s", "total_s"):
        representative.pop(internal_field)
    return representative


def first_stable_horizon(
    rows: Sequence[Mapping[str, object]],
    predicate,
) -> int:
    ordered = sorted(rows, key=lambda row: int(row["horizon"]))
    for candidate_index, candidate in enumerate(ordered):
        if all(predicate(row) for row in ordered[candidate_index:]):
            return int(candidate["horizon"])
    raise RuntimeError("No terminal comparison horizon was found.")


def annotate_stability(
    rows: list[dict[str, object]],
    max_horizon: int,
) -> None:
    by_alpha: Dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    for alpha_rows in by_alpha.values():
        ordered = sorted(alpha_rows, key=lambda row: int(row["horizon"]))
        terminal = next(
            row for row in ordered if int(row["horizon"]) == max_horizon
        )
        terminal_value = float(terminal["dynamic_regret_objective"])
        terminal_policy = terminal["reachable_policy_map_sha256"]
        for row in ordered:
            value_equal = isclose(
                float(row["dynamic_regret_objective"]),
                terminal_value,
                rel_tol=0.0,
                abs_tol=VALUE_TOLERANCE,
            )
            policy_equal = (
                row["reachable_policy_map_sha256"] == terminal_policy
            )
            row["comparison_max_horizon"] = max_horizon
            row["value_equal_to_max_horizon"] = int(value_equal)
            row["reachable_policy_equal_to_max_horizon"] = int(policy_equal)
            row["joint_value_policy_equal_to_max_horizon"] = int(
                value_equal and policy_equal
            )

        first_value = first_stable_horizon(
            ordered,
            lambda row: bool(row["value_equal_to_max_horizon"]),
        )
        first_policy = first_stable_horizon(
            ordered,
            lambda row: bool(row["reachable_policy_equal_to_max_horizon"]),
        )
        first_joint = first_stable_horizon(
            ordered,
            lambda row: bool(row["joint_value_policy_equal_to_max_horizon"]),
        )
        for row in ordered:
            row["first_value_stable_h_through_max"] = first_value
            row["first_policy_stable_h_through_max"] = first_policy
            row["first_joint_stable_h_through_max"] = first_joint
            row["joint_stabilization_observed_through_max"] = int(
                first_joint < max_horizon
            )

        for previous, current in zip(ordered, ordered[1:]):
            if (
                float(current["dynamic_regret_objective"])
                > float(previous["dynamic_regret_objective"])
                + VALUE_TOLERANCE
            ):
                raise RuntimeError(
                    "The Intel H-bounded objective increased with horizon."
                )


def compute_intel_rows(
    alphas: Sequence[float],
    max_horizon: int,
    repetitions: int,
    minimal_rank: int,
) -> list[dict[str, object]]:
    transition_system, _ = intel_lab_topological_pkwts()
    dfa = firefighting_dfa()
    prior = intel_two_mode_prior(
        transition_system
    ).explicit_prior_for_validation()
    rows = []
    configuration_count = len(alphas) * (max_horizon - minimal_rank + 1)
    completed = 0
    for alpha in alphas:
        for horizon in range(minimal_rank, max_horizon + 1):
            measurement = measured_intel_configuration(
                transition_system,
                dfa,
                prior,
                alpha,
                horizon,
                repetitions,
            )
            row = {
                "instance_id": "intel_lab_full_support",
                "alpha": alpha,
                "horizon": horizon,
                "minimal_robust_rank": minimal_rank,
                "horizon_slack_above_minimal_rank": horizon - minimal_rank,
                "world_count": len(prior),
                "dfa_state_count": len(dfa.states),
                **measurement,
                "runtime_scope": "local_diagnostic_not_submission_grade",
                "exact_world_enumeration": 1,
            }
            rows.append(row)
            completed += 1
            print(
                f"[PROGRESS] {completed}/{configuration_count} Intel configurations",
                flush=True,
            )
    annotate_stability(rows, max_horizon)
    return rows


def serialize_intel_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    float_fields = {
        "alpha",
        "dynamic_regret_objective",
        "exact_mean_regret",
        "exact_cvar95_regret",
        "exact_worst_regret",
        "initialization_median_s",
        "solve_median_s",
        "total_median_s",
        "total_min_s",
        "total_max_s",
    }
    serialized = []
    for row in rows:
        serialized.append(
            {
                field: (
                    format_number(float(row[field]))
                    if field in float_fields
                    else row[field]
                )
                for field in INTEL_FIELDS
            }
        )
    return serialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute benchmark robust-completion ranks and exact Intel "
            "horizon sensitivity."
        )
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=DEFAULT_MAX_HORIZON,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alphas = tuple(float(alpha) for alpha in args.alphas)
    if any(not 0.0 <= alpha < 1.0 for alpha in alphas):
        raise ValueError("Every alpha must satisfy 0 <= alpha < 1.")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least one.")

    rank_rows = compute_rank_rows()
    intel_rank_row = next(
        row
        for row in rank_rows
        if row["instance_id"] == "intel_lab_full_support"
    )
    minimal_intel_rank = int(intel_rank_row["minimal_robust_rank"])
    if args.max_horizon < max(13, minimal_intel_rank):
        raise ValueError(
            "--max-horizon must be at least 13 and no smaller than the "
            "Intel minimal robust rank."
        )
    intel_rows = compute_intel_rows(
        alphas,
        args.max_horizon,
        args.repetitions,
        minimal_intel_rank,
    )

    output_dir = args.output_dir.resolve()
    rank_path = output_dir / RANK_FILENAME
    intel_path = output_dir / INTEL_FILENAME
    write_csv(rank_path, RANK_FIELDS, rank_rows)
    write_csv(
        intel_path,
        INTEL_FIELDS,
        serialize_intel_rows(intel_rows),
    )

    observed = {
        float(row["alpha"]): int(row["first_joint_stable_h_through_max"])
        for row in intel_rows
        if int(row["horizon"]) == minimal_intel_rank
    }
    print(f"[OUTPUT] {rank_path} ({len(rank_rows)} rows)")
    print(f"[OUTPUT] {intel_path} ({len(intel_rows)} rows)")
    print(f"[RESULT] Intel minimal robust rank: {minimal_intel_rank}")
    print(
        "[RESULT] first joint value/policy stability through "
        f"H={args.max_horizon}: {observed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
