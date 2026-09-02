
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isclose
from typing import Dict, Mapping, Optional, Tuple

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
from stage1_3_risk import discrete_cvar


@dataclass
class DynamicCVaRResult:
    alpha: float
    policy: Dict[AgentNode, EnvNode]
    shifted_value: float
    dynamic_regret_value: float
    oracle_reference: float
    values: Dict[AgentNode, float]
    prior: Prior
    winning_nodes: int
    game_nodes: int
    iterations: int
    evaluation: dict
    policy_dynamic_regret_check: float


class DynamicCVaRRegretSolver:
    """
    Stage 1.4 time-consistent nested-CVaR regret solver.

    State:
        z = (x, q, K)

    Terminal shifted oracle gap:
        G(theta) = C_ref - J*(theta) >= 0
        C_ref = max_theta J*(theta)

    Terminal value at accepting information state z_F:
        T(z_F) = CVaR_alpha( G(Theta) | K_F )

    Bellman recursion at nonterminal robust-winning state z:
        V(z) = min_a [
            c(z,a) + CVaR_alpha( V(Z') | z,a )
        ]

    where accepting children use T(z_F) instead of a zero terminal value.

    The reported dynamic-regret objective is
        V(z0) - C_ref.

    Because nested conditional CVaR is used recursively, this criterion is
    time-consistent. It is intentionally different from the static/precommitment
    CVaR objective implemented in Stage 1.3.
    """

    def __init__(
        self,
        T: PKWTS,
        dfa: DFA,
        prior: Mapping[World, float],
        alpha: float,
        tol: float = 1e-10,
        max_iter: int = 10000,
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")

        full_prior = normalize_prior(T, prior)
        restricted_T = restrict_to_prior_support(T, full_prior)

        self.prior: Prior = full_prior
        self.T = restricted_T
        self.dfa = dfa
        self.alpha = float(alpha)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.game = Stage0RegretSolver(restricted_T, dfa)
        self.game.build_game()

        self.belief = BeliefModel(self.prior)
        self.winning = robust_winning_set(self.game)

        if self.game.start not in self.winning:
            raise ValueError(
                "No policy can satisfy the scLTL task for every world in supp(b0)."
            )

        self.oracle_costs = {w: self.game.oracle_cost(w) for w in self.prior}
        if any(v == inf for v in self.oracle_costs.values()):
            raise ValueError("At least one positive-probability world has infinite oracle cost.")

        self.oracle_reference = max(self.oracle_costs.values())
        self.oracle_gap = {
            w: self.oracle_reference - self.oracle_costs[w]
            for w in self.prior
        }

        self._terminal_cache: Dict[AgentNode, float] = {}

    # ---------- probability / terminal helpers ----------

    def branch_distribution(self, env: EnvNode) -> Dict[AgentNode, float]:
        probs = self.belief.observation_distribution(self.game, env)
        graph_children = {v for v, _ in self.game.adj[env]}
        if not set(probs).issubset(graph_children):
            raise RuntimeError("Bayesian branch is missing from the knowledge game.")
        return probs

    def terminal_shifted_value(self, a: AgentNode) -> float:
        if a.q not in self.dfa.accepting:
            raise ValueError("terminal_shifted_value called on nonaccepting node.")
        if a in self._terminal_cache:
            return self._terminal_cache[a]

        post = self.belief.posterior(self.game, a.K)
        worlds = list(post)
        val = discrete_cvar(
            [self.oracle_gap[w] for w in worlds],
            [post[w] for w in worlds],
            self.alpha,
        )
        self._terminal_cache[a] = val
        return val

    def _child_value(
        self,
        child: AgentNode,
        values: Mapping[AgentNode, float],
    ) -> float:
        if child.q in self.dfa.accepting:
            return self.terminal_shifted_value(child)
        if child not in values:
            raise RuntimeError(f"Missing value for nonterminal child {child}")
        return values[child]

    def _action_cost(self, env: EnvNode) -> float:
        edges = self.game.adj[env]
        if not edges:
            raise RuntimeError("Environment/action node has no physical successor.")
        weights = {float(w) for _, w in edges}
        if len(weights) != 1:
            raise RuntimeError(
                "A physical action should have the same immediate edge cost "
                "for all observation branches."
            )
        return next(iter(weights))

    def action_q(
        self,
        env: EnvNode,
        values: Mapping[AgentNode, float],
    ) -> float:
        probs = self.branch_distribution(env)
        child_vals = [self._child_value(ch, values) for ch in probs]
        child_probs = [probs[ch] for ch in probs]
        return self._action_cost(env) + discrete_cvar(
            child_vals, child_probs, self.alpha
        )

    # ---------- dynamic-risk Bellman solver ----------

    def solve(self) -> DynamicCVaRResult:
        vars_ = [
            a for a in self.game.agent_nodes
            if a in self.winning and a.q not in self.dfa.accepting
        ]

        # Nonnegative shifted SSP: monotone value iteration from zero.
        V: Dict[AgentNode, float] = {a: 0.0 for a in vars_}
        iterations = 0

        for k in range(1, self.max_iter + 1):
            V_new: Dict[AgentNode, float] = {}
            maxdiff = 0.0

            for a in vars_:
                actions = robust_actions(self.game, self.winning, a)
                if not actions:
                    raise RuntimeError(f"Winning state has no robust-winning action: {a}")

                qvals = [self.action_q(env, V) for env in actions]
                v = min(qvals)
                V_new[a] = v
                maxdiff = max(maxdiff, abs(v - V[a]))

            V = V_new
            iterations = k
            if maxdiff <= self.tol:
                break
        else:
            raise RuntimeError(
                f"Dynamic CVaR value iteration did not converge in {self.max_iter} iterations."
            )

        # Greedy Bellman policy.
        policy: Dict[AgentNode, EnvNode] = {}
        for a in vars_:
            candidates = []
            for env in robust_actions(self.game, self.winning, a):
                q = self.action_q(env, V)
                candidates.append((q, env.target, env))
            candidates.sort(key=lambda t: (t[0], t[1]))
            policy[a] = candidates[0][2]

        shifted_root = (
            self.terminal_shifted_value(self.game.start)
            if self.game.start.q in self.dfa.accepting
            else V[self.game.start]
        )
        dynamic_regret = shifted_root - self.oracle_reference

        evaluation = evaluate_policy(self.game, policy, self.prior)
        policy_check = self.evaluate_policy_dynamic_regret(policy)

        if not isclose(
            dynamic_regret, policy_check,
            rel_tol=1e-8, abs_tol=max(1e-8, self.tol * 100)
        ):
            raise RuntimeError(
                f"Bellman/policy dynamic-risk mismatch: "
                f"{dynamic_regret} vs {policy_check}"
            )

        return DynamicCVaRResult(
            alpha=self.alpha,
            policy=policy,
            shifted_value=shifted_root,
            dynamic_regret_value=dynamic_regret,
            oracle_reference=self.oracle_reference,
            values=V,
            prior=self.prior,
            winning_nodes=len(self.winning),
            game_nodes=len(self.game.adj),
            iterations=iterations,
            evaluation=evaluation,
            policy_dynamic_regret_check=policy_check,
        )

    # ---------- independent policy evaluation ----------

    def evaluate_policy_dynamic_regret(
        self,
        policy: Mapping[AgentNode, EnvNode],
    ) -> float:
        """
        Evaluate a fixed deterministic contingent policy by the same nested
        conditional-CVaR recursion, independently of the control minimization.
        """
        vars_ = [
            a for a in self.game.agent_nodes
            if a in self.winning and a.q not in self.dfa.accepting
        ]

        V: Dict[AgentNode, float] = {a: 0.0 for a in vars_}

        for _ in range(self.max_iter):
            V_new: Dict[AgentNode, float] = {}
            maxdiff = 0.0

            for a in vars_:
                if a not in policy:
                    # An unreachable state under this policy can still appear in the
                    # global graph. Give it +inf so it cannot accidentally be used.
                    V_new[a] = inf
                    continue
                env = policy[a]
                q = self.action_q(env, V)
                V_new[a] = q

                if V[a] != inf and q != inf:
                    maxdiff = max(maxdiff, abs(q - V[a]))
                elif V[a] != q:
                    maxdiff = inf

            V = V_new
            root = V.get(self.game.start, inf)
            if maxdiff <= self.tol:
                if root == inf:
                    raise RuntimeError("Fixed policy is not properly evaluable from start.")
                return root - self.oracle_reference

        raise RuntimeError("Fixed-policy dynamic CVaR evaluation did not converge.")

    # ---------- theorem-support diagnostics ----------

    def dynamic_min_positive_probability(self) -> float:
        """
        Minimum strictly positive local probability over:
          1) all observation distributions of robust-winning actions; and
          2) terminal posteriors at accepting winning states.

        If alpha >= 1 - this value, every local CVaR in the reachable finite
        model reduces to a local maximum. Under the fixed-world/noiseless
        knowledge model, the nested recursion then reduces to worst-world regret.
        """
        vals = []

        for a in self.game.agent_nodes:
            if a not in self.winning or a.q in self.dfa.accepting:
                continue
            for env in robust_actions(self.game, self.winning, a):
                for p in self.branch_distribution(env).values():
                    if p > 0.0:
                        vals.append(float(p))

        for a in self.game.accepting_nodes:
            if a not in self.winning:
                continue
            post = self.belief.posterior(self.game, a.K)
            for p in post.values():
                if p > 0.0:
                    vals.append(float(p))

        if not vals:
            return 1.0
        return min(vals)
