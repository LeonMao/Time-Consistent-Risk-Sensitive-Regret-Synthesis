from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import inf, isclose
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
for import_dir in (
    Path(__file__).resolve().parent,
    CORE_DIR,
):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage0_solver import AgentNode, DFA, EnvNode, PKWTS, World
from stage1_7_utils import (
    correlated_mixture_prior,
    eventually_goal_dfa,
    independent_prior,
    random_multishortcut,
)
from stage3_3_intel_lab_benchmark import (
    firefighting_dfa,
    intel_lab_topological_pkwts,
    intel_two_mode_prior,
)
from stage3_3_p2_static_precommitment import (
    ExactStaticCVaRPrecommitmentSolver,
    StaticPolicyCandidate,
)


DEFAULT_ALPHAS = (0.25, 0.50, 0.75, 0.90)
DEFAULT_RANDOM_SEEDS = 30
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reproduced_results"
RAW_FILENAME = "stage3_5_static_nested_mismatch_raw.csv"
SUMMARY_FILENAME = "stage3_5_static_nested_mismatch_summary.csv"
TOLERANCE = 1e-10


RAW_FIELDS = (
    "family",
    "instance_id",
    "non_hand_constructed",
    "seed",
    "prior_name",
    "alpha",
    "horizon",
    "world_count",
    "history_id",
    "history_depth",
    "information_state_id",
    "state_x",
    "state_q",
    "state_knowledge",
    "remaining_horizon",
    "past_cost",
    "reachable_world_count",
    "reachable_world_ids",
    "state_prior_probability",
    "static_committed_action",
    "static_conditional_reoptimized_action",
    "nested_conditional_action",
    "static_action_mismatch",
    "static_vs_nested_action_difference",
    "static_committed_conditional_cvar_regret",
    "static_reoptimized_conditional_cvar_regret",
    "static_conditional_cvar_improvement",
    "static_value_mismatch",
    "nested_conditional_dynamic_regret_value",
    "exact_world_enumeration",
)


SUMMARY_FIELDS = (
    "family",
    "instance_id",
    "non_hand_constructed",
    "seed",
    "prior_name",
    "alpha",
    "horizon",
    "world_count",
    "prior_probability_sum",
    "static_candidates_considered",
    "static_root_action",
    "nested_root_action",
    "root_action_difference",
    "reachable_policy_difference",
    "reachable_policy_difference_probability",
    "common_reachable_information_state_count",
    "common_reachable_action_difference_count",
    "static_reachable_decision_history_count",
    "static_action_mismatch_state_count",
    "static_action_mismatch_probability",
    "static_value_mismatch_state_count",
    "static_value_mismatch_probability",
    "max_static_conditional_cvar_improvement",
    "static_root_cvar_regret",
    "nested_root_dynamic_regret_value",
    "exact_world_enumeration",
)


@dataclass(frozen=True)
class InstanceSpec:
    family: str
    instance_id: str
    non_hand_constructed: bool
    seed: int | None
    prior_name: str
    horizon: int
    transition_system: PKWTS
    dfa: DFA
    prior: Mapping[World, float]


@dataclass(frozen=True)
class ReachableHistory:
    history_id: str
    depth: int
    state: AgentNode
    remaining_horizon: int
    past_cost: float
    worlds: Tuple[World, ...]


def format_float(value: float) -> str:
    return f"{value:.12g}"


def knowledge_text(state: AgentNode) -> str:
    if not state.K:
        return "-"
    return ";".join(f"{name}:{pattern}" for name, pattern in sorted(state.K))


def state_id(state: AgentNode, remaining_horizon: int) -> str:
    return (
        f"x={state.x}|q={state.q}|K={knowledge_text(state)}|"
        f"h={remaining_horizon}"
    )


def world_id(transition_system: PKWTS, world: World) -> str:
    uncertain = [
        f"{name}:{world[transition_system.state_index[name]]}"
        for name in transition_system.states
        if len(transition_system.patterns[name]) > 1
    ]
    return ";".join(uncertain) if uncertain else "deterministic"


def policy_future_costs(
    solver: ExactStaticCVaRPrecommitmentSolver,
    policy: Mapping[Tuple[AgentNode, int], EnvNode],
    state: AgentNode,
    remaining_horizon: int,
) -> Dict[World, float]:
    costs: Dict[World, float] = {}
    for world in solver.compatible_worlds(state.K):
        current = state
        budget = remaining_horizon
        future_cost = 0.0
        while current.q not in solver.A.accepting:
            if budget <= 0:
                raise RuntimeError("Committed policy exceeded its finite horizon.")
            key = (current, budget)
            if key not in policy:
                raise RuntimeError(f"Committed policy is undefined at {state_id(*key)}.")
            action = policy[key]
            future_cost += float(solver.T.weights[(action.x, action.target)])
            current = solver.child_for_world(action, world)
            budget -= 1
        costs[world] = future_cost
    return costs


def reachable_histories(
    solver: ExactStaticCVaRPrecommitmentSolver,
    policy: Mapping[Tuple[AgentNode, int], EnvNode],
) -> Tuple[ReachableHistory, ...]:
    start_worlds = tuple(solver.worlds)
    stack = [
        ReachableHistory(
            history_id=solver.T.x0,
            depth=0,
            state=solver.start,
            remaining_horizon=solver.horizon,
            past_cost=0.0,
            worlds=start_worlds,
        )
    ]
    reached = []

    while stack:
        history = stack.pop()
        if history.state.q in solver.A.accepting:
            continue
        if history.remaining_horizon <= 0:
            raise RuntimeError("A reachable decision history has no remaining budget.")
        reached.append(history)
        action = policy[(history.state, history.remaining_horizon)]
        branch_worlds: Dict[AgentNode, list[World]] = {}
        for world in history.worlds:
            child = solver.child_for_world(action, world)
            branch_worlds.setdefault(child, []).append(world)

        edge_cost = float(solver.T.weights[(action.x, action.target)])
        ordered_branches = sorted(
            branch_worlds.items(),
            key=lambda item: state_id(item[0], history.remaining_horizon - 1),
            reverse=True,
        )
        for child, worlds in ordered_branches:
            observation = knowledge_text(child)
            stack.append(
                ReachableHistory(
                    history_id=(
                        f"{history.history_id}>{action.target}[K={observation}]"
                    ),
                    depth=history.depth + 1,
                    state=child,
                    remaining_horizon=history.remaining_horizon - 1,
                    past_cost=history.past_cost + edge_cost,
                    worlds=tuple(worlds),
                )
            )

    return tuple(reached)


def simulate_policy_world(
    solver: ExactStaticCVaRPrecommitmentSolver,
    policy: Mapping[Tuple[AgentNode, int], EnvNode],
    world: World,
) -> Tuple[Tuple[str, ...], Dict[Tuple[AgentNode, int], str]]:
    current = solver.start
    budget = solver.horizon
    actions = []
    reached_actions: Dict[Tuple[AgentNode, int], str] = {}
    while current.q not in solver.A.accepting:
        if budget <= 0:
            raise RuntimeError("Policy execution exceeded its finite horizon.")
        key = (current, budget)
        action = policy[key]
        actions.append(action.target)
        reached_actions[key] = action.target
        current = solver.child_for_world(action, world)
        budget -= 1
    return tuple(actions), reached_actions


def build_instances(random_seed_count: int) -> Iterable[InstanceSpec]:
    intel_system, _ = intel_lab_topological_pkwts()
    intel_prior = intel_two_mode_prior(intel_system).explicit_prior_for_validation()
    for horizon in (9, 11):
        yield InstanceSpec(
            family="intel_lab_topological",
            instance_id=f"intel_lab_H{horizon}",
            non_hand_constructed=False,
            seed=None,
            prior_name="two_mode_full_support",
            horizon=horizon,
            transition_system=intel_system,
            dfa=firefighting_dfa(),
            prior=intel_prior,
        )

    for seed in range(random_seed_count):
        transition_system = random_multishortcut(m=3, seed=seed)
        dfa = eventually_goal_dfa()
        for probability_open in (0.10, 0.30, 0.50, 0.70, 0.90):
            prior = independent_prior(
                transition_system,
                p_open=probability_open,
            ).explicit_prior_for_validation()
            yield InstanceSpec(
                family="random_multishortcut_m3",
                instance_id=f"random_multishortcut_m3_seed{seed:03d}",
                non_hand_constructed=True,
                seed=seed,
                prior_name=f"independent_p{probability_open:.2f}",
                horizon=5,
                transition_system=transition_system,
                dfa=dfa,
                prior=prior,
            )

        correlated_prior = correlated_mixture_prior(
            transition_system,
            modes=2,
            seed=seed,
            center=0.45,
            spread=0.65,
        ).explicit_prior_for_validation()
        yield InstanceSpec(
            family="random_multishortcut_m3",
            instance_id=f"random_multishortcut_m3_seed{seed:03d}",
            non_hand_constructed=True,
            seed=seed,
            prior_name="correlated_two_mode",
            horizon=5,
            transition_system=transition_system,
            dfa=dfa,
            prior=correlated_prior,
        )


def analyze_configuration(
    spec: InstanceSpec,
    alpha: float,
) -> Tuple[list[dict[str, object]], dict[str, object]]:
    static_solver = ExactStaticCVaRPrecommitmentSolver(
        spec.transition_system,
        spec.dfa,
        spec.prior,
        alpha,
        spec.horizon,
    )
    static_result = static_solver.solve()
    nested_solver = LazyHorizonDynamicCVaRSolver(
        spec.transition_system,
        spec.dfa,
        spec.prior,
        alpha,
        spec.horizon,
    )
    nested_result = nested_solver.solve()

    static_root_action = static_result.policy[(static_solver.start, spec.horizon)].target
    nested_root_action = nested_result.policy[(nested_solver.start, spec.horizon)].target
    histories = reachable_histories(static_solver, static_result.policy)
    mismatch_worlds: set[World] = set()
    value_mismatch_worlds: set[World] = set()
    raw_rows: list[dict[str, object]] = []
    max_improvement = 0.0

    for history in histories:
        key = (history.state, history.remaining_horizon)
        committed_action = static_result.policy[key]
        conditional_static = static_solver.solve_from(
            history.state,
            history.remaining_horizon,
            history.past_cost,
        )
        conditional_action = conditional_static.policy[key]
        committed_candidate = StaticPolicyCandidate(
            future_costs=policy_future_costs(
                static_solver,
                static_result.policy,
                history.state,
                history.remaining_horizon,
            ),
            policy={},
        )
        committed_cvar, _, _, _ = static_solver.evaluate_candidate(
            history.state,
            committed_candidate,
            history.past_cost,
        )
        improvement = committed_cvar - conditional_static.cvar_regret
        if improvement < -TOLERANCE:
            raise RuntimeError(
                "Conditional static reoptimization is worse than its committed continuation."
            )
        if abs(improvement) <= TOLERANCE:
            improvement = 0.0
        max_improvement = max(max_improvement, improvement)

        nested_shifted_value = nested_solver.budget_value(
            history.state,
            history.remaining_horizon,
        )
        if nested_shifted_value == inf:
            raise RuntimeError("A static-reachable state is infeasible for nested CVaR.")
        nested_action = nested_solver.budget_policy[key]
        action_mismatch = committed_action.target != conditional_action.target
        value_mismatch = improvement > TOLERANCE
        static_nested_difference = committed_action.target != nested_action.target
        if action_mismatch:
            mismatch_worlds.update(history.worlds)
        if value_mismatch:
            value_mismatch_worlds.update(history.worlds)

        reachable_probability = sum(spec.prior[world] for world in history.worlds)
        if reachable_probability <= 0:
            raise RuntimeError("A reported mismatch row is not prior-reachable.")
        raw_rows.append(
            {
                "family": spec.family,
                "instance_id": spec.instance_id,
                "non_hand_constructed": int(spec.non_hand_constructed),
                "seed": "" if spec.seed is None else spec.seed,
                "prior_name": spec.prior_name,
                "alpha": format_float(alpha),
                "horizon": spec.horizon,
                "world_count": len(spec.prior),
                "history_id": history.history_id,
                "history_depth": history.depth,
                "information_state_id": state_id(
                    history.state, history.remaining_horizon
                ),
                "state_x": history.state.x,
                "state_q": history.state.q,
                "state_knowledge": knowledge_text(history.state),
                "remaining_horizon": history.remaining_horizon,
                "past_cost": format_float(history.past_cost),
                "reachable_world_count": len(history.worlds),
                "reachable_world_ids": "|".join(
                    world_id(spec.transition_system, world)
                    for world in history.worlds
                ),
                "state_prior_probability": format_float(reachable_probability),
                "static_committed_action": committed_action.target,
                "static_conditional_reoptimized_action": conditional_action.target,
                "nested_conditional_action": nested_action.target,
                "static_action_mismatch": int(action_mismatch),
                "static_vs_nested_action_difference": int(static_nested_difference),
                "static_committed_conditional_cvar_regret": format_float(
                    committed_cvar
                ),
                "static_reoptimized_conditional_cvar_regret": format_float(
                    conditional_static.cvar_regret
                ),
                "static_conditional_cvar_improvement": format_float(improvement),
                "static_value_mismatch": int(value_mismatch),
                "nested_conditional_dynamic_regret_value": format_float(
                    nested_shifted_value - nested_solver.oracle_reference
                ),
                "exact_world_enumeration": 1,
            }
        )

    static_reached_actions: Dict[Tuple[AgentNode, int], str] = {}
    nested_reached_actions: Dict[Tuple[AgentNode, int], str] = {}
    policy_difference_worlds: set[World] = set()
    for world in static_solver.worlds:
        static_path, static_actions = simulate_policy_world(
            static_solver,
            static_result.policy,
            world,
        )
        nested_path, nested_actions = simulate_policy_world(
            static_solver,
            nested_result.policy,
            world,
        )
        static_reached_actions.update(static_actions)
        nested_reached_actions.update(nested_actions)
        if static_path != nested_path:
            policy_difference_worlds.add(world)

    common_states = set(static_reached_actions) & set(nested_reached_actions)
    common_action_differences = sum(
        static_reached_actions[key] != nested_reached_actions[key]
        for key in common_states
    )
    mismatch_probability = sum(spec.prior[world] for world in mismatch_worlds)
    value_mismatch_probability = sum(
        spec.prior[world] for world in value_mismatch_worlds
    )
    policy_difference_probability = sum(
        spec.prior[world] for world in policy_difference_worlds
    )

    summary = {
        "family": spec.family,
        "instance_id": spec.instance_id,
        "non_hand_constructed": int(spec.non_hand_constructed),
        "seed": "" if spec.seed is None else spec.seed,
        "prior_name": spec.prior_name,
        "alpha": format_float(alpha),
        "horizon": spec.horizon,
        "world_count": len(spec.prior),
        "prior_probability_sum": format_float(sum(spec.prior.values())),
        "static_candidates_considered": static_result.candidates_considered,
        "static_root_action": static_root_action,
        "nested_root_action": nested_root_action,
        "root_action_difference": int(static_root_action != nested_root_action),
        "reachable_policy_difference": int(bool(policy_difference_worlds)),
        "reachable_policy_difference_probability": format_float(
            policy_difference_probability
        ),
        "common_reachable_information_state_count": len(common_states),
        "common_reachable_action_difference_count": common_action_differences,
        "static_reachable_decision_history_count": len(histories),
        "static_action_mismatch_state_count": sum(
            int(row["static_action_mismatch"]) for row in raw_rows
        ),
        "static_action_mismatch_probability": format_float(mismatch_probability),
        "static_value_mismatch_state_count": sum(
            int(row["static_value_mismatch"]) for row in raw_rows
        ),
        "static_value_mismatch_probability": format_float(
            value_mismatch_probability
        ),
        "max_static_conditional_cvar_improvement": format_float(max_improvement),
        "static_root_cvar_regret": format_float(static_result.cvar_regret),
        "nested_root_dynamic_regret_value": format_float(
            nested_result.dynamic_regret_value
        ),
        "exact_world_enumeration": 1,
    }
    return raw_rows, summary


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact Stage 3.5 static-precommitment versus nested-CVaR "
            "mismatch sweep."
        )
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=DEFAULT_RANDOM_SEEDS,
        help="Number of deterministic random multishortcut seeds.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="CVaR confidence levels in [0,1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the raw and summary CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.random_seeds < 1:
        raise ValueError("--random-seeds must be at least 1.")
    alphas = tuple(float(alpha) for alpha in args.alphas)
    if any(not 0.0 <= alpha < 1.0 for alpha in alphas):
        raise ValueError("Every alpha must satisfy 0 <= alpha < 1.")

    instances = tuple(build_instances(args.random_seeds))
    configuration_count = len(instances) * len(alphas)
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for index, spec in enumerate(instances, start=1):
        for alpha in alphas:
            configuration_raw, configuration_summary = analyze_configuration(
                spec,
                alpha,
            )
            raw_rows.extend(configuration_raw)
            summary_rows.append(configuration_summary)
        if index == 1 or index % 20 == 0 or index == len(instances):
            completed = index * len(alphas)
            print(f"[PROGRESS] {completed}/{configuration_count} configurations")

    output_dir = args.output_dir.resolve()
    raw_path = output_dir / RAW_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    write_csv(raw_path, RAW_FIELDS, raw_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    random_mismatch_rows = [
        row
        for row in summary_rows
        if int(row["non_hand_constructed"]) == 1
        and float(row["static_action_mismatch_probability"]) > TOLERANCE
    ]
    random_policy_difference_rows = [
        row
        for row in summary_rows
        if int(row["non_hand_constructed"]) == 1
        and float(row["reachable_policy_difference_probability"]) > TOLERANCE
    ]
    probability_checks = [
        isclose(float(row["prior_probability_sum"]), 1.0, abs_tol=TOLERANCE)
        for row in summary_rows
    ]
    if not all(probability_checks):
        raise RuntimeError("At least one explicit prior does not sum to one.")

    print(f"[OUTPUT] {raw_path} ({len(raw_rows)} rows)")
    print(f"[OUTPUT] {summary_path} ({len(summary_rows)} rows)")
    print(
        "[RESULT] non-hand-constructed static reoptimization mismatches: "
        f"{len(random_mismatch_rows)} configurations"
    )
    print(
        "[RESULT] non-hand-constructed static/nested reachable-policy "
        f"differences: {len(random_policy_difference_rows)} configurations"
    )
    if random_mismatch_rows:
        print("[CLAIM STATUS] POSITIVE_EXACT_MISMATCH_EVIDENCE")
    else:
        print("[CLAIM STATUS] NEGATIVE_RESULT_NARROW_CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
