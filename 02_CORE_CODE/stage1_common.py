from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, FrozenSet, Mapping, Tuple

from stage0_solver import AgentNode, EnvNode, GameNode, Knowledge, PKWTS, Stage0RegretSolver, World

Prior = Dict[World, float]
Posterior = Dict[World, float]


def normalize_prior(T: PKWTS, prior: Mapping[World, float], tol: float = 1e-12) -> Prior:
    """Validate and normalize a prior over possible worlds.

    Zero-probability worlds are removed from the support. The returned support
    is the set on which robust scLTL satisfaction will be enforced in Stage 1.2.
    """
    all_worlds = set(T.all_worlds())
    out: Prior = {}
    for world, p in prior.items():
        if world not in all_worlds:
            raise ValueError(f"Prior contains a world not admitted by the PK-WTS: {world}")
        p = float(p)
        if p < -tol:
            raise ValueError("Prior probabilities must be nonnegative.")
        if p > tol:
            out[world] = p
    total = sum(out.values())
    if total <= tol:
        raise ValueError("Prior must assign positive mass to at least one possible world.")
    return {w: p / total for w, p in out.items()}


def restrict_to_prior_support(T: PKWTS, prior: Mapping[World, float]) -> PKWTS:
    """Create a Stage-0-compatible PK-WTS view restricted to positive prior support.

    The frozen Stage-0 solver is not modified; a new PKWTS object is constructed
    with allowed_worlds equal to supp(b0).
    """
    support = tuple(prior.keys())
    return PKWTS(
        states=T.states,
        x0=T.x0,
        patterns=T.patterns,
        weights=T.weights,
        labels=T.labels,
        allowed_worlds=support,
    )


@dataclass(frozen=True)
class BeliefModel:
    """Exact Bayesian belief induced by a fixed-world prior and noiseless observations."""

    prior: Mapping[World, float]

    def posterior(self, solver: Stage0RegretSolver, K: Knowledge) -> Posterior:
        compatible = set(solver.compatible_worlds(K))
        masses = {w: p for w, p in self.prior.items() if w in compatible and p > 0.0}
        z = sum(masses.values())
        if z <= 0.0:
            raise RuntimeError("Knowledge state is inconsistent with the prior support.")
        return {w: p / z for w, p in masses.items()}

    def observation_distribution(
        self,
        solver: Stage0RegretSolver,
        env: EnvNode,
    ) -> Dict[AgentNode, float]:
        """P(next agent node | current knowledge, chosen physical target)."""
        post = self.posterior(solver, env.K)
        probs: Dict[AgentNode, float] = {}

        for world, p in post.items():
            pidx = solver.T.world_pattern_index(world, env.target)
            K2 = solver.update_knowledge(env.K, env.target, pidx)
            q2 = solver.A.step(env.q, solver.T.labels[env.target])
            child = AgentNode(env.target, q2, K2)
            probs[child] = probs.get(child, 0.0) + p

        # Numerical cleanup / validation.
        total = sum(probs.values())
        if not isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            probs = {v: p / total for v, p in probs.items()}
        return probs


def robust_winning_set(solver: Stage0RegretSolver) -> FrozenSet[GameNode]:
    """Reachability winning set for robust scLTL satisfaction over prior support.

    Agent node is winning if it has at least one action leading to a winning
    environment node. Environment node is winning if *all* observation branches
    lead to winning agent nodes. This is the standard reachability attractor on
    the bipartite knowledge game.
    """
    winning = set(solver.accepting_nodes)

    changed = True
    while changed:
        changed = False

        # Nature/environment vertices: all possible observations must remain winning.
        for e in solver.env_nodes:
            if e in winning:
                continue
            succ = [v for v, _ in solver.adj[e]]
            if succ and all(v in winning for v in succ):
                winning.add(e)
                changed = True

        # Agent vertices: at least one robust-winning action must exist.
        for a in solver.agent_nodes:
            if a in winning:
                continue
            succ = [v for v, _ in solver.adj[a]]
            if any(v in winning for v in succ):
                winning.add(a)
                changed = True

    return frozenset(winning)


def robust_actions(solver: Stage0RegretSolver, winning: FrozenSet[GameNode], a: AgentNode):
    """Environment/action nodes that preserve robust eventual task satisfaction."""
    return tuple(v for v, _ in solver.adj[a] if isinstance(v, EnvNode) and v in winning)


def expected_oracle_cost(prior: Mapping[World, float], oracle_costs: Mapping[World, float]) -> float:
    return sum(prior[w] * oracle_costs[w] for w in prior)


def evaluate_policy(
    solver: Stage0RegretSolver,
    policy: Mapping[AgentNode, EnvNode],
    prior: Mapping[World, float],
):
    """World-wise and Bayesian evaluation of a deterministic contingent policy."""
    costs: Dict[World, float] = {}
    regrets: Dict[World, float] = {}
    expected_cost = 0.0
    expected_regret = 0.0
    oracle_costs = {w: solver.oracle_cost(w) for w in prior}

    for w, p in prior.items():
        cost, _ = solver.simulate_policy(policy, w)
        regret = cost - oracle_costs[w]
        costs[w] = cost
        regrets[w] = regret
        expected_cost += p * cost
        expected_regret += p * regret

    oracle_mean = expected_oracle_cost(prior, oracle_costs)
    return {
        "costs": costs,
        "regrets": regrets,
        "oracle_costs": oracle_costs,
        "expected_cost": expected_cost,
        "expected_regret": expected_regret,
        "expected_oracle_cost": oracle_mean,
        "equivalence_residual": expected_regret - (expected_cost - oracle_mean),
    }
