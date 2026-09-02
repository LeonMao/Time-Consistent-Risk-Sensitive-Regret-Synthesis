from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from math import inf, isclose, isfinite
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for import_dir in (EXPERIMENT_DIR, CORE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from factored_belief import MixtureProductPrior, ProductComponent
from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage0_solver import AgentNode, DFA, EnvNode, PKWTS, World
from stage1_3_risk import discrete_cvar
from stage3_3_intel_lab_benchmark import (
    firefighting_dfa,
    intel_lab_topological_pkwts,
)
from stage3_5_horizon_sensitivity import (
    format_number,
    reachable_policy_map,
    write_csv,
)


DEFAULT_ALPHAS = (0.0, 0.25, 0.50, 0.75, 0.90)
DEFAULT_HORIZON = 11
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reproduced_results"
OUTPUT_FILENAME = "stage3_5_prior_robustness.csv"
LOCAL_OFFSETS = (-0.10, 0.10)
MIXTURE_GOOD_WEIGHTS = (0.35, 0.45, 0.65, 0.75)
CONTAMINATION_MASSES = (0.001, 0.005, 0.01, 0.02, 0.05)
VALUE_TOLERANCE = 1e-10


OUTPUT_FIELDS = (
    "scenario_id",
    "scenario_class",
    "perturbation_target",
    "perturbation_value",
    "alpha",
    "horizon",
    "nominal_support_world_count",
    "true_support_world_count",
    "support_expanded",
    "nominal_zero_mass_true_world_count",
    "nominal_zero_mass_true_prior_mass",
    "total_variation_distance_from_nominal",
    "true_good_mode_weight",
    "local_state",
    "local_open_offset",
    "contamination_mass",
    "nominal_policy_root_action",
    "resynthesized_root_action",
    "root_action_changed",
    "nominal_policy_map_sha256",
    "resynthesized_policy_map_sha256",
    "nominal_policy_evaluation_status",
    "nominal_policy_failure_reasons",
    "nominal_policy_unsupported_world_count",
    "nominal_policy_nested_objective_true_prior",
    "resynthesized_nested_objective_true_prior",
    "nested_objective_improvement_from_resynthesis",
    "nominal_policy_mean_regret_true_prior",
    "resynthesized_mean_regret_true_prior",
    "nominal_policy_cvar95_regret_true_prior",
    "resynthesized_cvar95_regret_true_prior",
    "nominal_policy_worst_regret_true_prior",
    "resynthesized_worst_regret_true_prior",
    "nominal_policy_all_true_worlds_satisfied",
    "nominal_policy_satisfied_true_prior_mass",
    "resynthesized_all_true_worlds_satisfied",
    "resynthesized_satisfied_true_prior_mass",
    "hard_guarantee_retained_by_nominal_policy",
    "exact_world_enumeration",
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    scenario_class: str
    perturbation_target: str
    perturbation_value: float
    transition_system: PKWTS
    prior: Mapping[World, float]
    support_expanded: bool
    true_good_mode_weight: float | None = None
    local_state: str | None = None
    local_open_offset: float | None = None
    contamination_mass: float = 0.0


@dataclass(frozen=True)
class NominalSolution:
    policy: Mapping[Tuple[AgentNode, int], EnvNode]
    root_action: str
    policy_hash: str
    dynamic_objective: float
    mean_regret: float
    cvar95_regret: float
    worst_regret: float


@dataclass(frozen=True)
class FixedPolicyEvaluation:
    status: str
    failure_reasons: str
    unsupported_world_count: int
    dynamic_objective: float | None
    mean_regret: float | None
    cvar95_regret: float | None
    worst_regret: float | None
    all_worlds_satisfied: bool
    satisfied_prior_mass: float


def normalize_explicit_prior(prior: Mapping[World, float]) -> Dict[World, float]:
    positive = {world: float(mass) for world, mass in prior.items() if mass > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        raise ValueError("Prior has no positive mass.")
    return {world: mass / total for world, mass in positive.items()}


def intel_two_mode_explicit_prior(
    transition_system: PKWTS,
    good_weight: float = 0.55,
    local_offsets: Mapping[str, float] | None = None,
) -> Dict[World, float]:
    if not 0.0 < good_weight < 1.0:
        raise ValueError("The good-mode weight must be strictly between zero and one.")
    offsets = dict(local_offsets or {})
    good = {}
    restricted = {}
    for state in transition_system.states:
        if len(transition_system.patterns[state]) <= 1:
            continue
        offset = float(offsets.get(state, 0.0))
        good_open = 0.78 + offset
        restricted_open = 0.18 + offset
        if not 0.0 < good_open < 1.0 or not 0.0 < restricted_open < 1.0:
            raise ValueError(f"Local offset leaves the probability simplex at {state}.")
        good[state] = (1.0 - good_open, good_open)
        restricted[state] = (1.0 - restricted_open, restricted_open)
    factored = MixtureProductPrior(
        transition_system,
        (
            ProductComponent(good_weight, good),
            ProductComponent(1.0 - good_weight, restricted),
        ),
    )
    return factored.explicit_prior_for_validation()


def expanded_d0_jam_system(transition_system: PKWTS) -> PKWTS:
    patterns = dict(transition_system.patterns)
    patterns["d0"] = tuple(patterns["d0"]) + (frozenset(),)
    return PKWTS(
        states=transition_system.states,
        x0=transition_system.x0,
        patterns=patterns,
        weights=transition_system.weights,
        labels=transition_system.labels,
    )


def contamination_world(transition_system: PKWTS) -> World:
    values = [0] * len(transition_system.states)
    values[transition_system.state_index["d0"]] = 2
    return tuple(values)


def contaminated_prior(
    nominal_prior: Mapping[World, float],
    transition_system: PKWTS,
    contamination_mass: float,
) -> Dict[World, float]:
    if not 0.0 < contamination_mass < 1.0:
        raise ValueError("Contamination mass must be strictly between zero and one.")
    prior = {
        world: (1.0 - contamination_mass) * probability
        for world, probability in nominal_prior.items()
    }
    prior[contamination_world(transition_system)] = contamination_mass
    return normalize_explicit_prior(prior)


def total_variation_distance(
    nominal_prior: Mapping[World, float],
    true_prior: Mapping[World, float],
) -> float:
    worlds = set(nominal_prior) | set(true_prior)
    distance = 0.5 * sum(
        abs(float(nominal_prior.get(world, 0.0)) - float(true_prior.get(world, 0.0)))
        for world in worlds
    )
    return 0.0 if distance < 1e-15 else distance


def build_scenarios(
    transition_system: PKWTS,
    nominal_prior: Mapping[World, float],
) -> Tuple[Scenario, ...]:
    scenarios = [
        Scenario(
            scenario_id="nominal",
            scenario_class="nominal",
            perturbation_target="none",
            perturbation_value=0.0,
            transition_system=transition_system,
            prior=nominal_prior,
            support_expanded=False,
            true_good_mode_weight=0.55,
        )
    ]
    for good_weight in MIXTURE_GOOD_WEIGHTS:
        scenarios.append(
            Scenario(
                scenario_id=f"mixture_good_{good_weight:.2f}",
                scenario_class="prior_weight_error",
                perturbation_target="good_mode_weight",
                perturbation_value=good_weight - 0.55,
                transition_system=transition_system,
                prior=intel_two_mode_explicit_prior(
                    transition_system,
                    good_weight=good_weight,
                ),
                support_expanded=False,
                true_good_mode_weight=good_weight,
            )
        )
    local_states = tuple(
        state
        for state in transition_system.states
        if len(transition_system.patterns[state]) > 1
    )
    for state in local_states:
        for offset in LOCAL_OFFSETS:
            sign = "plus" if offset > 0 else "minus"
            scenarios.append(
                Scenario(
                    scenario_id=f"local_{state}_{sign}_{abs(offset):.2f}",
                    scenario_class="prior_weight_error",
                    perturbation_target=f"{state}_open_probability",
                    perturbation_value=offset,
                    transition_system=transition_system,
                    prior=intel_two_mode_explicit_prior(
                        transition_system,
                        local_offsets={state: offset},
                    ),
                    support_expanded=False,
                    true_good_mode_weight=0.55,
                    local_state=state,
                    local_open_offset=offset,
                )
            )
    expanded_system = expanded_d0_jam_system(transition_system)
    for mass in CONTAMINATION_MASSES:
        scenarios.append(
            Scenario(
                scenario_id=f"support_d0_jam_{mass:.3f}",
                scenario_class="missing_support_error",
                perturbation_target="d0_jammed_dead_end_world",
                perturbation_value=mass,
                transition_system=expanded_system,
                prior=contaminated_prior(
                    nominal_prior,
                    expanded_system,
                    mass,
                ),
                support_expanded=True,
                contamination_mass=mass,
            )
        )
    return tuple(scenarios)


def child_for_world(
    transition_system: PKWTS,
    dfa: DFA,
    action: EnvNode,
    world: World,
) -> AgentNode:
    pattern_index = transition_system.world_pattern_index(world, action.target)
    knowledge = action.K
    if len(transition_system.patterns[action.target]) > 1:
        knowledge = frozenset(set(knowledge) | {(action.target, pattern_index)})
    child_q = dfa.step(action.q, transition_system.labels[action.target])
    return AgentNode(action.target, child_q, knowledge)


def simulate_fixed_policy_world(
    solver: LazyHorizonDynamicCVaRSolver,
    policy: Mapping[Tuple[AgentNode, int], EnvNode],
    world: World,
) -> Tuple[float | None, AgentNode, str]:
    state = solver.start
    remaining = solver.horizon
    cost = 0.0
    while state.q not in solver.A.accepting:
        if remaining <= 0:
            return None, state, "horizon_exhausted"
        action = policy.get((state, remaining))
        if action is None:
            return None, state, "undefined_out_of_support_history"
        if action.target not in solver.current_successors(state):
            return None, state, "invalid_action_under_true_world"
        cost += solver.action_cost(action)
        state = child_for_world(solver.T, solver.A, action, world)
        remaining -= 1
    return cost, state, "accepted"


def evaluate_fixed_policy(
    solver: LazyHorizonDynamicCVaRSolver,
    policy: Mapping[Tuple[AgentNode, int], EnvNode],
) -> FixedPolicyEvaluation:
    cache: Dict[Tuple[AgentNode, int], float] = {}

    def nested_value(state: AgentNode, remaining: int) -> float:
        if state.q in solver.A.accepting:
            return solver.terminal_value(state)
        if remaining <= 0:
            return inf
        key = (state, remaining)
        if key in cache:
            return cache[key]
        action = policy.get(key)
        if action is None or action.target not in solver.current_successors(state):
            cache[key] = inf
            return inf
        probabilities = solver.branches(action)
        child_values = [nested_value(child, remaining - 1) for child in probabilities]
        if any(not isfinite(value) for value in child_values):
            cache[key] = inf
            return inf
        value = solver.action_cost(action) + discrete_cvar(
            child_values,
            [probabilities[child] for child in probabilities],
            solver.alpha,
        )
        cache[key] = value
        return value

    shifted_value = nested_value(solver.start, solver.horizon)
    dynamic_objective = (
        shifted_value - solver.oracle_reference
        if isfinite(shifted_value)
        else None
    )
    costs: Dict[World, float] = {}
    reasons: Counter[str] = Counter()
    satisfied_mass = 0.0
    for world, probability in solver.prior.items():
        cost, terminal, status = simulate_fixed_policy_world(solver, policy, world)
        if status == "accepted" and cost is not None and terminal.q in solver.A.accepting:
            costs[world] = cost
            satisfied_mass += probability
        else:
            reasons[status] += 1

    all_satisfied = len(costs) == len(solver.prior)
    if all_satisfied:
        regrets = {
            world: costs[world] - solver.oracle_costs[world]
            for world in solver.prior
        }
        probabilities = [solver.prior[world] for world in solver.prior]
        regret_values = [regrets[world] for world in solver.prior]
        mean_regret = sum(
            probability * regret
            for probability, regret in zip(probabilities, regret_values)
        )
        cvar95_regret = discrete_cvar(regret_values, probabilities, 0.95)
        worst_regret = max(regret_values)
        status = "complete"
    else:
        mean_regret = None
        cvar95_regret = None
        worst_regret = None
        status = "undefined_on_expanded_support"

    reason_text = ";".join(
        f"{reason}:{count}" for reason, count in sorted(reasons.items())
    )
    return FixedPolicyEvaluation(
        status=status,
        failure_reasons=reason_text,
        unsupported_world_count=sum(reasons.values()),
        dynamic_objective=dynamic_objective,
        mean_regret=mean_regret,
        cvar95_regret=cvar95_regret,
        worst_regret=worst_regret,
        all_worlds_satisfied=all_satisfied,
        satisfied_prior_mass=satisfied_mass,
    )


def exact_metrics(
    solver: LazyHorizonDynamicCVaRSolver,
    regrets: Mapping[World, float],
) -> Tuple[float, float, float]:
    probabilities = [solver.prior[world] for world in solver.worlds]
    values = [regrets[world] for world in solver.worlds]
    mean_regret = sum(
        probability * regret
        for probability, regret in zip(probabilities, values)
    )
    return (
        mean_regret,
        discrete_cvar(values, probabilities, 0.95),
        max(values),
    )


def solve_nominal(
    transition_system: PKWTS,
    dfa: DFA,
    prior: Mapping[World, float],
    alpha: float,
    horizon: int,
) -> NominalSolution:
    solver = LazyHorizonDynamicCVaRSolver(
        transition_system,
        dfa,
        prior,
        alpha,
        horizon,
    )
    result = solver.solve()
    _, policy_hash, _ = reachable_policy_map(solver, result.policy)
    mean_regret, cvar95_regret, worst_regret = exact_metrics(
        solver,
        result.policy_world_regrets,
    )
    return NominalSolution(
        policy=dict(result.policy),
        root_action=result.policy[(solver.start, horizon)].target,
        policy_hash=policy_hash,
        dynamic_objective=result.dynamic_regret_value,
        mean_regret=mean_regret,
        cvar95_regret=cvar95_regret,
        worst_regret=worst_regret,
    )


def optional_number(value: float | None) -> str:
    return "" if value is None else format_number(value)


def compute_rows(
    alphas: Sequence[float],
    horizon: int,
) -> list[dict[str, object]]:
    transition_system, _ = intel_lab_topological_pkwts()
    dfa = firefighting_dfa()
    nominal_prior = intel_two_mode_explicit_prior(transition_system)
    scenarios = build_scenarios(transition_system, nominal_prior)
    nominal_solutions = {
        alpha: solve_nominal(
            transition_system,
            dfa,
            nominal_prior,
            alpha,
            horizon,
        )
        for alpha in alphas
    }
    rows = []
    total = len(alphas) * len(scenarios)
    completed = 0
    for alpha in alphas:
        nominal = nominal_solutions[alpha]
        for scenario in scenarios:
            solver = LazyHorizonDynamicCVaRSolver(
                scenario.transition_system,
                dfa,
                scenario.prior,
                alpha,
                horizon,
            )
            result = solver.solve()
            _, resynthesized_hash, _ = reachable_policy_map(solver, result.policy)
            resynthesized_mean, resynthesized_cvar95, resynthesized_worst = exact_metrics(
                solver,
                result.policy_world_regrets,
            )
            nominal_evaluation = evaluate_fixed_policy(solver, nominal.policy)
            nominal_zero_worlds = [
                world
                for world in solver.prior
                if nominal_prior.get(world, 0.0) == 0.0
            ]
            nominal_zero_mass = sum(
                solver.prior[world] for world in nominal_zero_worlds
            )
            improvement = (
                None
                if nominal_evaluation.dynamic_objective is None
                else nominal_evaluation.dynamic_objective
                - result.dynamic_regret_value
            )
            if improvement is not None and improvement < -VALUE_TOLERANCE:
                raise RuntimeError("A resynthesized optimum is worse than the fixed nominal policy.")
            if not all(
                solver.simulate_budget_policy_world(world)[1].q in dfa.accepting
                for world in solver.worlds
            ):
                raise RuntimeError("A resynthesized policy violated hard satisfaction.")
            row = {
                "scenario_id": scenario.scenario_id,
                "scenario_class": scenario.scenario_class,
                "perturbation_target": scenario.perturbation_target,
                "perturbation_value": format_number(scenario.perturbation_value),
                "alpha": format_number(alpha),
                "horizon": horizon,
                "nominal_support_world_count": len(nominal_prior),
                "true_support_world_count": len(solver.prior),
                "support_expanded": int(scenario.support_expanded),
                "nominal_zero_mass_true_world_count": len(nominal_zero_worlds),
                "nominal_zero_mass_true_prior_mass": format_number(nominal_zero_mass),
                "total_variation_distance_from_nominal": format_number(
                    total_variation_distance(nominal_prior, solver.prior)
                ),
                "true_good_mode_weight": optional_number(
                    scenario.true_good_mode_weight
                ),
                "local_state": scenario.local_state or "",
                "local_open_offset": optional_number(scenario.local_open_offset),
                "contamination_mass": format_number(scenario.contamination_mass),
                "nominal_policy_root_action": nominal.root_action,
                "resynthesized_root_action": result.policy[
                    (solver.start, horizon)
                ].target,
                "root_action_changed": int(
                    nominal.root_action
                    != result.policy[(solver.start, horizon)].target
                ),
                "nominal_policy_map_sha256": nominal.policy_hash,
                "resynthesized_policy_map_sha256": resynthesized_hash,
                "nominal_policy_evaluation_status": nominal_evaluation.status,
                "nominal_policy_failure_reasons": nominal_evaluation.failure_reasons,
                "nominal_policy_unsupported_world_count": (
                    nominal_evaluation.unsupported_world_count
                ),
                "nominal_policy_nested_objective_true_prior": optional_number(
                    nominal_evaluation.dynamic_objective
                ),
                "resynthesized_nested_objective_true_prior": format_number(
                    result.dynamic_regret_value
                ),
                "nested_objective_improvement_from_resynthesis": optional_number(
                    improvement
                ),
                "nominal_policy_mean_regret_true_prior": optional_number(
                    nominal_evaluation.mean_regret
                ),
                "resynthesized_mean_regret_true_prior": format_number(
                    resynthesized_mean
                ),
                "nominal_policy_cvar95_regret_true_prior": optional_number(
                    nominal_evaluation.cvar95_regret
                ),
                "resynthesized_cvar95_regret_true_prior": format_number(
                    resynthesized_cvar95
                ),
                "nominal_policy_worst_regret_true_prior": optional_number(
                    nominal_evaluation.worst_regret
                ),
                "resynthesized_worst_regret_true_prior": format_number(
                    resynthesized_worst
                ),
                "nominal_policy_all_true_worlds_satisfied": int(
                    nominal_evaluation.all_worlds_satisfied
                ),
                "nominal_policy_satisfied_true_prior_mass": format_number(
                    nominal_evaluation.satisfied_prior_mass
                ),
                "resynthesized_all_true_worlds_satisfied": 1,
                "resynthesized_satisfied_true_prior_mass": "1",
                "hard_guarantee_retained_by_nominal_policy": int(
                    nominal_evaluation.all_worlds_satisfied
                ),
                "exact_world_enumeration": 1,
            }
            if scenario.scenario_class == "nominal":
                checks = (
                    isclose(
                        float(row["nominal_policy_nested_objective_true_prior"]),
                        nominal.dynamic_objective,
                        rel_tol=0.0,
                        abs_tol=VALUE_TOLERANCE,
                    ),
                    isclose(
                        float(row["nominal_policy_mean_regret_true_prior"]),
                        nominal.mean_regret,
                        rel_tol=0.0,
                        abs_tol=VALUE_TOLERANCE,
                    ),
                    isclose(
                        float(row["nominal_policy_cvar95_regret_true_prior"]),
                        nominal.cvar95_regret,
                        rel_tol=0.0,
                        abs_tol=VALUE_TOLERANCE,
                    ),
                    isclose(
                        float(row["nominal_policy_worst_regret_true_prior"]),
                        nominal.worst_regret,
                        rel_tol=0.0,
                        abs_tol=VALUE_TOLERANCE,
                    ),
                )
                if not all(checks):
                    raise RuntimeError("Nominal fixed-policy evaluation failed its identity check.")
            rows.append(row)
            completed += 1
            print(
                f"[PROGRESS] {completed}/{total} prior-robustness configurations",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Intel prior-weight perturbations and one missing-support "
            "contamination world by exact enumeration."
        )
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
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
    if args.horizon < 9:
        raise ValueError("The Intel horizon must be at least its robust rank 9.")
    rows = compute_rows(alphas, args.horizon)
    output_path = args.output_dir.resolve() / OUTPUT_FILENAME
    write_csv(output_path, OUTPUT_FIELDS, rows)

    prior_weight_rows = [
        row for row in rows if row["scenario_class"] == "prior_weight_error"
    ]
    contamination_rows = [
        row for row in rows if row["scenario_class"] == "missing_support_error"
    ]
    print(f"[PASS] Wrote {len(rows)} exact rows to {output_path}")
    print(
        "[SUMMARY] prior-weight root changes="
        f"{sum(int(row['root_action_changed']) for row in prior_weight_rows)}/"
        f"{len(prior_weight_rows)}"
    )
    print(
        "[SUMMARY] missing-support nominal hard failures="
        f"{sum(1 - int(row['hard_guarantee_retained_by_nominal_policy']) for row in contamination_rows)}/"
        f"{len(contamination_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
