
from __future__ import annotations

from dataclasses import dataclass
from math import log2
import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from factored_belief import MixtureProductPrior, ProductComponent
from stage0_solver import AgentNode, DFA, PKWTS, World
from stage1_3_risk import discrete_cvar


def eventually_goal_dfa(goal_label: str = "goal") -> DFA:
    def tr(q, label):
        if q == "qF":
            return "qF"
        return "qF" if goal_label in label else "qN"
    return DFA(("qN", "qF"), "qN", frozenset({"qF"}), tr)


def ordered_goal_dfa(k: int) -> DFA:
    """Monitor for ordered finite task: g1 then g2 ... then gk."""
    if k < 1:
        raise ValueError("k must be >=1")
    states = tuple([f"q{i}" for i in range(k)] + ["qF"])

    def tr(q, label):
        if q == "qF":
            return "qF"
        i = int(q[1:])
        if f"g{i+1}" in label:
            return "qF" if i + 1 == k else f"q{i+1}"
        return q

    return DFA(states, "q0", frozenset({"qF"}), tr)


def random_multishortcut(
    m: int,
    seed: int,
    safe_cost: float = 30.0,
    entry_range: Tuple[float, float] = (0.8, 1.5),
    back_range: Tuple[float, float] = (0.8, 1.8),
    exit_range: Tuple[float, float] = (1.5, 8.0),
) -> PKWTS:
    """
    One-hub family with m binary monotone shortcuts.
    closed pattern = {s}; open pattern = {s,g}.
    """
    rng = random.Random(seed)
    unknowns = tuple(f"u{i}" for i in range(m))
    states = ("s",) + unknowns + ("g",)

    patterns = {
        "s": (frozenset(set(unknowns) | {"g"}),),
        "g": (frozenset(),),
    }
    weights = {("s", "g"): float(safe_cost)}
    labels = {x: frozenset() for x in states}
    labels["g"] = frozenset({"goal"})

    # Sorted exit costs create a range of shortcut qualities but randomize entries/backtracks.
    exit_costs = sorted(rng.uniform(*exit_range) for _ in range(m))
    for i, u in enumerate(unknowns):
        patterns[u] = (
            frozenset({"s"}),
            frozenset({"s", "g"}),
        )
        weights[("s", u)] = rng.uniform(*entry_range)
        weights[(u, "s")] = rng.uniform(*back_range)
        weights[(u, "g")] = exit_costs[i]

    return PKWTS(states, "s", patterns, weights, labels)


def ordered_stage_benchmark(
    k: int,
    shortcuts_per_stage: int,
    seed: int,
) -> Tuple[PKWTS, DFA]:
    """
    k ordered milestones. Each stage has a known safe edge from hub h_i to g_{i+1}
    and several binary monotone shortcuts u_i_j that may reach the same milestone.
    After milestone g_i (except the last), a known edge leads to the next hub.
    """
    if k < 1 or shortcuts_per_stage < 1:
        raise ValueError("k and shortcuts_per_stage must be >=1")

    rng = random.Random(seed)
    hubs = tuple(f"h{i}" for i in range(k))
    goals = tuple(f"g{i+1}" for i in range(k))
    unknowns = tuple(
        f"u{i}_{j}"
        for i in range(k)
        for j in range(shortcuts_per_stage)
    )
    states = hubs + goals + unknowns

    patterns = {}
    weights = {}
    labels = {x: frozenset() for x in states}

    for i, h in enumerate(hubs):
        stage_unknowns = tuple(f"u{i}_{j}" for j in range(shortcuts_per_stage))
        patterns[h] = (frozenset(set(stage_unknowns) | {goals[i]}),)
        weights[(h, goals[i])] = 8.0 + rng.uniform(0.0, 3.0)

        exits = sorted(rng.uniform(1.5, 5.0) for _ in stage_unknowns)
        for j, u in enumerate(stage_unknowns):
            patterns[u] = (
                frozenset({h}),
                frozenset({h, goals[i]}),
            )
            weights[(h, u)] = rng.uniform(0.8, 1.5)
            weights[(u, h)] = rng.uniform(0.8, 1.6)
            weights[(u, goals[i])] = exits[j]

        labels[goals[i]] = frozenset({f"g{i+1}"})
        if i < k - 1:
            patterns[goals[i]] = (frozenset({hubs[i+1]}),)
            weights[(goals[i], hubs[i+1])] = 1.0
        else:
            patterns[goals[i]] = (frozenset(),)

    return PKWTS(states, hubs[0], patterns, weights, labels), ordered_goal_dfa(k)


def independent_prior(
    T: PKWTS,
    p_open: float,
    seed: int = 0,
    jitter: float = 0.0,
) -> MixtureProductPrior:
    rng = random.Random(seed)
    marg = {}
    for x in T.states:
        if len(T.patterns[x]) > 1:
            p = min(0.95, max(0.05, p_open + rng.uniform(-jitter, jitter)))
            marg[x] = (1.0 - p, p)
    return MixtureProductPrior(T, [ProductComponent(1.0, marg)])


def correlated_mixture_prior(
    T: PKWTS,
    modes: int,
    seed: int = 0,
    center: float = 0.40,
    spread: float = 0.55,
) -> MixtureProductPrior:
    """
    Latent-mode correlated prior. Mode open probabilities are spread from
    center-spread/2 to center+spread/2 and receive random positive weights.
    """
    if modes < 1:
        raise ValueError("modes must be >=1")
    rng = random.Random(seed)
    raw_w = [rng.uniform(0.5, 1.5) for _ in range(modes)]
    z = sum(raw_w)
    weights = [w / z for w in raw_w]

    if modes == 1:
        probs = [center]
    else:
        lo = max(0.05, center - spread / 2)
        hi = min(0.95, center + spread / 2)
        probs = [lo + (hi - lo) * i / (modes - 1) for i in range(modes)]

    comps = []
    for c in range(modes):
        marg = {}
        for x in T.states:
            if len(T.patterns[x]) > 1:
                # Slight variable-specific perturbation preserves the latent mode trend.
                p = min(0.97, max(0.03, probs[c] + rng.uniform(-0.04, 0.04)))
                marg[x] = (1.0 - p, p)
        comps.append(ProductComponent(weights[c], marg))
    return MixtureProductPrior(T, comps)


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * log2(p) + (1 - p) * log2(1 - p))


def sample_world(prior: MixtureProductPrior, rng: random.Random) -> World:
    # Sample latent component.
    r = rng.random()
    cum = 0.0
    mode = len(prior.components) - 1
    for i, comp in enumerate(prior.components):
        cum += comp.weight
        if r <= cum:
            mode = i
            break

    comp = prior.components[mode]
    vals = []
    for x in prior.T.states:
        if len(prior.T.patterns[x]) == 1:
            vals.append(0)
            continue
        probs = comp.marginals[x]
        rr = rng.random()
        cc = 0.0
        chosen = len(probs) - 1
        for k, p in enumerate(probs):
            cc += p
            if rr <= cc:
                chosen = k
                break
        vals.append(chosen)
    return tuple(vals)


def simulate_factored_budget_policy(solver, result, world: World):
    a = solver.start
    h = solver.horizon
    cost = 0.0

    while True:
        if a.q in solver.A.accepting:
            return cost, a
        if h <= 0:
            raise RuntimeError("Policy failed within H.")

        env = result.policy[(a, h)]
        pidx = world[solver.T.state_index[env.target]]
        K2 = env.K
        if len(solver.T.patterns[env.target]) > 1:
            K2 = frozenset(set(K2) | {(env.target, pidx)})
        q2 = solver.A.step(env.q, solver.T.labels[env.target])
        a = AgentNode(env.target, q2, K2)
        cost += solver.action_cost(env)
        h -= 1


def empirical_policy_metrics(
    solver,
    result,
    samples: int,
    seed: int,
    alpha_eval: float = 0.95,
):
    rng = random.Random(seed)
    regrets = []
    costs = []
    sat = 0

    for _ in range(samples):
        w = sample_world(solver.prior, rng)
        c, terminal = simulate_factored_budget_policy(solver, result, w)
        jstar = solver.oracle._oracle_cost_tuple(w)
        costs.append(c)
        regrets.append(c - jstar)
        if terminal.q in solver.A.accepting:
            sat += 1

    probs = [1.0 / samples] * samples
    return {
        "mean_cost": sum(costs) / samples,
        "mean_regret": sum(regrets) / samples,
        "cvar95_regret": discrete_cvar(regrets, probs, alpha_eval),
        "max_sample_regret": max(regrets),
        "satisfaction_rate": sat / samples,
    }
