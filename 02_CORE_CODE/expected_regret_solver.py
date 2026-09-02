from __future__ import annotations

from dataclasses import dataclass
from math import inf, isinf
from typing import Dict, Mapping, Optional

import numpy as np
from scipy.optimize import linprog

from stage0_solver import AgentNode, DFA, EnvNode, PKWTS, Stage0RegretSolver, World
from stage1_common import (
    BeliefModel,
    Prior,
    evaluate_policy,
    normalize_prior,
    restrict_to_prior_support,
    robust_actions,
    robust_winning_set,
)


@dataclass
class ExpectedRegretResult:
    policy: Dict[AgentNode, EnvNode]
    expected_cost: float
    expected_regret: float
    expected_oracle_cost: float
    values: Dict[AgentNode, float]
    prior: Prior
    winning_nodes: int
    game_nodes: int
    evaluation: dict


class BayesianExpectedRegretSolver:
    """Stage 1.2 Bayesian expected-regret comparator.

    Key theorem used by the implementation:

        argmin_pi E[J_pi(theta) - J*(theta)]
        = argmin_pi E[J_pi(theta)],

    because E[J*(theta)] is independent of pi for a fixed prior.

    The solver therefore minimizes Bayesian expected physical cost *within the
    robust scLTL-winning strategy set*. Expected regret is then computed exactly
    by subtracting the prior-weighted clairvoyant benchmark.

    The frozen Stage-0 code is reused only as a graph/oracle/simulation engine;
    it is not modified.
    """

    def __init__(self, T: PKWTS, dfa: DFA, prior: Mapping[World, float]):
        full_prior = normalize_prior(T, prior)
        restricted_T = restrict_to_prior_support(T, full_prior)

        self.prior: Prior = full_prior
        self.T = restricted_T
        self.dfa = dfa
        self.game = Stage0RegretSolver(restricted_T, dfa)
        self.game.build_game()
        self.belief = BeliefModel(self.prior)
        self.winning = robust_winning_set(self.game)

        if self.game.start not in self.winning:
            raise ValueError(
                "No policy can satisfy the scLTL task for every world in supp(b0)."
            )

    def branch_distribution(self, env: EnvNode):
        probs = self.belief.observation_distribution(self.game, env)
        # Sanity: game branches and positive-probability Bayesian branches must agree.
        graph_children = {v for v, _ in self.game.adj[env]}
        if not set(probs).issubset(graph_children):
            raise RuntimeError("Bayesian branch is missing from the knowledge game.")
        return probs

    def _action_q_expression(self, env: EnvNode, index: Dict[AgentNode, int]):
        """Return immediate expected cost and coefficients of child V values."""
        probs = self.branch_distribution(env)
        immediate = 0.0
        coeff: Dict[AgentNode, float] = {}

        edge_weight = {v: w for v, w in self.game.adj[env]}
        for child, p in probs.items():
            immediate += p * edge_weight[child]
            if child.q not in self.dfa.accepting:
                coeff[child] = coeff.get(child, 0.0) + p
        return immediate, coeff

    def solve(self) -> ExpectedRegretResult:
        # Variables only for robust-winning, nonterminal agent nodes.
        vars_ = [
            a for a in self.game.agent_nodes
            if a in self.winning and a.q not in self.dfa.accepting
        ]
        index = {a: i for i, a in enumerate(vars_)}
        n = len(vars_)

        if n == 0:
            values = {}
            policy = {}
        else:
            # Shortest stochastic path LP:
            #   maximize sum_a V(a)
            #   s.t. V(a) <= c(a,u) + E[V(a')], for every robust action u.
            # scipy.linprog minimizes, hence c = -1.
            A_ub = []
            b_ub = []

            for a in vars_:
                actions = robust_actions(self.game, self.winning, a)
                if not actions:
                    raise RuntimeError(f"Winning agent node has no winning action: {a}")

                for env in actions:
                    immediate, coeff = self._action_q_expression(env, index)
                    row = np.zeros(n, dtype=float)
                    row[index[a]] = 1.0
                    for child, p in coeff.items():
                        if child not in index:
                            # A nonterminal child of a robust winning action should be a variable.
                            raise RuntimeError(f"Unexpected nonterminal child outside LP: {child}")
                        row[index[child]] -= p
                    A_ub.append(row)
                    b_ub.append(immediate)

            res = linprog(
                c=-np.ones(n, dtype=float),
                A_ub=np.asarray(A_ub, dtype=float),
                b_ub=np.asarray(b_ub, dtype=float),
                bounds=[(0.0, None)] * n,
                method="highs",
            )
            if not res.success:
                raise RuntimeError(f"Expected-cost LP failed: {res.message}")

            values = {a: float(res.x[index[a]]) for a in vars_}

            # Extract Bellman-minimizing robust action at each variable state.
            policy: Dict[AgentNode, EnvNode] = {}
            tol = 1e-8
            for a in vars_:
                candidates = []
                for env in robust_actions(self.game, self.winning, a):
                    immediate, coeff = self._action_q_expression(env, index)
                    q = immediate + sum(p * values[ch] for ch, p in coeff.items())
                    candidates.append((q, repr(env), env))
                candidates.sort(key=lambda z: (z[0], z[1]))
                best_q, _, best_env = candidates[0]
                # LP variable is the minimum of its action Q values.
                if abs(values[a] - best_q) > max(tol, tol * max(1.0, abs(best_q))):
                    raise RuntimeError(
                        f"Bellman/LP consistency failure at {a}: V={values[a]}, minQ={best_q}"
                    )
                policy[a] = best_env

        evaluation = evaluate_policy(self.game, policy, self.prior)
        expected_cost = evaluation["expected_cost"]
        expected_regret = evaluation["expected_regret"]
        expected_oracle = evaluation["expected_oracle_cost"]

        # Initial LP value must equal simulated Bayesian expected cost.
        if self.game.start.q not in self.dfa.accepting:
            v0 = values[self.game.start]
            if abs(v0 - expected_cost) > 1e-7 * max(1.0, abs(expected_cost)):
                raise RuntimeError(
                    f"Initial LP value {v0} != simulated expected cost {expected_cost}."
                )

        if abs(evaluation["equivalence_residual"]) > 1e-9:
            raise RuntimeError("Expected-regret identity failed numerically.")

        return ExpectedRegretResult(
            policy=policy,
            expected_cost=expected_cost,
            expected_regret=expected_regret,
            expected_oracle_cost=expected_oracle,
            values=values,
            prior=self.prior,
            winning_nodes=len(self.winning),
            game_nodes=len(self.game.adj),
            evaluation=evaluation,
        )
