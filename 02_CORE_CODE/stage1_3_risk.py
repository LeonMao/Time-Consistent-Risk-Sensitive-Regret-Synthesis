
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from stage0_solver import AgentNode, EnvNode, Stage0RegretSolver, World
from stage1_common import normalize_prior, evaluate_policy


def _normalize_discrete(values: Sequence[float], probs: Sequence[float]):
    if len(values) == 0 or len(values) != len(probs):
        raise ValueError("values and probs must be nonempty and have the same length.")
    v = np.asarray(values, dtype=float)
    p = np.asarray(probs, dtype=float)
    if np.any(p < -1e-14):
        raise ValueError("Probabilities must be nonnegative.")
    s = float(np.sum(p))
    if s <= 0:
        raise ValueError("Probability mass must be positive.")
    p = p / s
    return v, p


def discrete_var(values: Sequence[float], probs: Sequence[float], alpha: float) -> float:
    """Lower alpha-quantile VaR_alpha for a finite loss distribution."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    v, p = _normalize_discrete(values, probs)
    order = np.argsort(v, kind="stable")
    cum = 0.0
    for i in order:
        cum += float(p[i])
        if cum + 1e-15 >= alpha:
            return float(v[i])
    return float(np.max(v))


def discrete_cvar(
    values: Sequence[float],
    probs: Sequence[float],
    alpha: float,
) -> float:
    """Rockafellar-Uryasev CVaR for a finite loss distribution.

    CVaR_alpha(Z) = min_eta eta + E[(Z-eta)_+] / (1-alpha).

    For a finite distribution, an optimizer eta can be chosen from the support.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    v, p = _normalize_discrete(values, probs)
    denom = 1.0 - alpha

    best = float("inf")
    for eta in np.unique(v):
        obj = float(eta + np.sum(p * np.maximum(v - eta, 0.0)) / denom)
        if obj < best:
            best = obj
    return best


def cvar_risk_envelope(
    values: Sequence[float],
    probs: Sequence[float],
    alpha: float,
) -> Tuple[float, np.ndarray]:
    """Dual risk-envelope representation of finite-distribution CVaR.

    CVaR_alpha(Z) =
        max_q sum_i q_i Z_i
        s.t. sum_i q_i = 1,
             0 <= q_i <= p_i/(1-alpha).

    Returns (optimal CVaR value, one maximizing distorted distribution q).
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha < 1.")
    v, p = _normalize_discrete(values, probs)

    ub = p / (1.0 - alpha)
    res = linprog(
        c=-v,
        A_eq=np.ones((1, len(v))),
        b_eq=np.array([1.0]),
        bounds=[(0.0, float(u)) for u in ub],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"CVaR risk-envelope LP failed: {res.message}")

    return float(-res.fun), np.asarray(res.x, dtype=float)


def cvar_from_regret_map(
    regrets: Mapping[World, float],
    prior: Mapping[World, float],
    alpha: float,
) -> float:
    worlds = list(prior)
    return discrete_cvar(
        [regrets[w] for w in worlds],
        [prior[w] for w in worlds],
        alpha,
    )


@dataclass
class StaticCVaREvaluation:
    alpha: float
    cvar_regret: float
    var_regret: float
    expected_regret: float
    worst_regret: float
    expected_cost: float
    distorted_world_weights: Dict[World, float]
    evaluation: dict


def evaluate_policy_cvar(
    solver: Stage0RegretSolver,
    policy: Mapping[AgentNode, EnvNode],
    prior: Mapping[World, float],
    alpha: float,
) -> StaticCVaREvaluation:
    prior_n = normalize_prior(solver.T, prior)
    ev = evaluate_policy(solver, policy, prior_n)

    worlds = list(prior_n)
    regrets = [ev["regrets"][w] for w in worlds]
    probs = [prior_n[w] for w in worlds]

    cvar = discrete_cvar(regrets, probs, alpha)
    var = discrete_var(regrets, probs, alpha)
    dual_value, q = cvar_risk_envelope(regrets, probs, alpha)

    if not isclose(cvar, dual_value, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(
            f"CVaR primal/dual mismatch: primal={cvar}, envelope={dual_value}"
        )

    return StaticCVaREvaluation(
        alpha=alpha,
        cvar_regret=cvar,
        var_regret=var,
        expected_regret=ev["expected_regret"],
        worst_regret=max(ev["regrets"].values()),
        expected_cost=ev["expected_cost"],
        distorted_world_weights={w: float(q[i]) for i, w in enumerate(worlds)},
        evaluation=ev,
    )


def finite_support_minimax_threshold(prior: Mapping[World, float]) -> float:
    """alpha_crit = 1 - min positive prior mass.

    For every finite loss vector on this support, CVaR_alpha equals max loss
    whenever alpha >= alpha_crit.
    """
    positive = [float(p) for p in prior.values() if p > 0.0]
    if not positive:
        raise ValueError("Prior support is empty.")
    return 1.0 - min(positive)
