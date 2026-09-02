from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
for import_dir in (EXPERIMENT_DIR, CORE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from stage0_solver import AgentNode, World
from stage1_3_risk import discrete_cvar


def weighted_var(values: list[float], probabilities: list[float], alpha: float) -> float:
    if len(values) != len(probabilities) or not values:
        raise ValueError("values and probabilities must be nonempty and aligned")
    cumulative = 0.0
    for value, probability in sorted(zip(values, probabilities)):
        cumulative += probability
        if cumulative >= alpha - 1e-15:
            return float(value)
    return float(max(values))


def simulate_budget_policy(solver, policy: Mapping, world: World):
    agent = solver.start
    horizon = solver.horizon
    cost = 0.0
    actions: list[str] = []

    while True:
        if agent.q in solver.A.accepting:
            return cost, agent, tuple(actions)
        if horizon <= 0:
            raise RuntimeError("Policy failed to satisfy the task within H.")

        env = policy[(agent, horizon)]
        actions.append(str(env.target))
        pattern_index = world[solver.T.state_index[env.target]]
        knowledge = env.K
        if len(solver.T.patterns[env.target]) > 1:
            knowledge = frozenset(set(knowledge) | {(env.target, pattern_index)})
        q_next = solver.A.step(env.q, solver.T.labels[env.target])
        agent = AgentNode(env.target, q_next, knowledge)
        cost += solver.action_cost(env)
        horizon -= 1


def exact_policy_metrics(solver, policy: Mapping) -> dict[str, float | int]:
    explicit_prior = solver.prior.explicit_prior_for_validation()
    costs: list[float] = []
    regrets: list[float] = []
    probabilities: list[float] = []
    satisfied = True

    for world, probability in explicit_prior.items():
        cost, terminal, _ = simulate_budget_policy(solver, policy, world)
        regret = cost - solver.oracle._oracle_cost_tuple(world)
        costs.append(float(cost))
        regrets.append(float(regret))
        probabilities.append(float(probability))
        satisfied = satisfied and terminal.q in solver.A.accepting

    total_probability = sum(probabilities)
    if not isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Prior mass is {total_probability}, expected 1.")
    return {
        "worlds": len(explicit_prior),
        "mean_cost": sum(p * value for p, value in zip(probabilities, costs)),
        "mean_regret": sum(p * value for p, value in zip(probabilities, regrets)),
        "var95_regret": weighted_var(regrets, probabilities, 0.95),
        "cvar95_regret": discrete_cvar(regrets, probabilities, 0.95),
        "worst_regret": max(regrets),
        "worst_cost": max(costs),
        "satisfaction_all_worlds": int(satisfied),
    }


def action_map(policy: Mapping) -> dict[object, str]:
    return {key: str(env.target) for key, env in policy.items()}

