
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isclose
import heapq
from typing import Dict, FrozenSet, Mapping, Optional, Set, Tuple

from stage0_solver import AgentNode, DFA, EnvNode, Knowledge, PKWTS
from stage1_3_risk import discrete_cvar
from factored_belief import MixtureProductPrior
from symbolic_oracle import MonotoneBinaryOracleDiagram


@dataclass
class FactoredDynamicCVaRResult:
    alpha: float
    horizon: int
    shifted_value: float
    dynamic_regret_value: float
    oracle_reference: float
    policy: Dict[Tuple[AgentNode, int], EnvNode]
    generated_agent_states: int
    generated_observation_branches: int
    value_state_budgets: int
    pruned_actions: int
    symbolic_oracle_nodes: int
    symbolic_oracle_collapses: int
    oracle_shortest_path_calls: int
    conceptual_world_upper_bound: int


class FactoredLazyHorizonDynamicCVaRSolver:
    """
    Exact horizon-bounded dynamic-CVaR regret solver without explicit world
    enumeration.

    Prior:
      mixture of product categorical factors over unknown PK-WTS states.

    Topology/oracle restriction:
      binary monotone unknown patterns, handled by MonotoneBinaryOracleDiagram.

    The solver is exact over Pi_H under the supplied factored prior.
    """

    def __init__(
        self,
        T: PKWTS,
        dfa: DFA,
        prior: MixtureProductPrior,
        alpha: float,
        horizon: int,
        use_lower_bound_pruning: bool = True,
    ):
        if prior.T is not T:
            # Identity is convenient but not essential; require matching state tuple.
            if prior.T.states != T.states:
                raise ValueError("Factored prior does not match PK-WTS.")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")

        self.T = T
        self.A = dfa
        self.prior = prior
        self.alpha = float(alpha)
        self.horizon = int(horizon)
        self.use_lower_bound_pruning = bool(use_lower_bound_pruning)

        q0 = self.A.step(self.A.q_init, self.T.labels[self.T.x0])
        self.start = AgentNode(self.T.x0, q0, frozenset())

        self.oracle = MonotoneBinaryOracleDiagram(T, dfa, prior)
        self.oracle_reference = self.oracle.oracle_reference

        self._actions_cache: Dict[AgentNode, Tuple[EnvNode, ...]] = {}
        self._branches_cache: Dict[EnvNode, Dict[AgentNode, float]] = {}
        self._can_cache: Dict[Tuple[AgentNode, int], bool] = {}
        self._value_cache: Dict[Tuple[AgentNode, int], float] = {}
        self._terminal_cache: Dict[AgentNode, float] = {}
        self._optimistic_lb_cache: Dict[AgentNode, float] = {}

        self.policy: Dict[Tuple[AgentNode, int], EnvNode] = {}

        self.generated_agent_states: Set[AgentNode] = {self.start}
        self.generated_observation_branches = 0
        self.value_state_budgets = 0
        self.pruned_actions = 0

    # ---------- implicit knowledge transitions ----------

    def current_successors(self, a: AgentNode):
        pats = self.T.patterns[a.x]
        if len(pats) == 1:
            return pats[0]
        kd = dict(a.K)
        if a.x not in kd:
            raise RuntimeError(f"Unknown current state {a.x} has not been observed.")
        return pats[kd[a.x]]

    def actions(self, a: AgentNode) -> Tuple[EnvNode, ...]:
        if a in self._actions_cache:
            return self._actions_cache[a]
        if a.q in self.A.accepting:
            self._actions_cache[a] = tuple()
            return tuple()

        out = tuple(
            EnvNode(a.x, a.q, a.K, y)
            for y in sorted(self.current_successors(a))
        )
        self._actions_cache[a] = out
        return out

    def branches(self, env: EnvNode) -> Dict[AgentNode, float]:
        if env in self._branches_cache:
            return self._branches_cache[env]

        dist = self.prior.observation_distribution(env.K, env.target)
        out: Dict[AgentNode, float] = {}

        for pidx, p in dist.items():
            if p <= 0:
                continue
            K2 = self.prior.condition(env.K, env.target, pidx)
            q2 = self.A.step(env.q, self.T.labels[env.target])
            child = AgentNode(env.target, q2, K2)
            out[child] = out.get(child, 0.0) + p

        z = sum(out.values())
        if z <= 0:
            raise RuntimeError("No posterior-consistent observation branch.")
        out = {ch: p / z for ch, p in out.items()}

        self._branches_cache[env] = out
        self.generated_agent_states.update(out)
        self.generated_observation_branches += len(out)
        return out

    def action_cost(self, env: EnvNode) -> float:
        return float(self.T.weights[(env.x, env.target)])

    # ---------- robust H-step feasibility ----------

    def can_accept_within(self, a: AgentNode, h: int) -> bool:
        key = (a, h)
        if key in self._can_cache:
            return self._can_cache[key]

        if a.q in self.A.accepting:
            self._can_cache[key] = True
            return True
        if h <= 0:
            self._can_cache[key] = False
            return False

        for env in self.actions(a):
            children = self.branches(env)
            if children and all(self.can_accept_within(ch, h - 1) for ch in children):
                self._can_cache[key] = True
                return True

        self._can_cache[key] = False
        return False

    # ---------- terminal oracle risk ----------

    def terminal_value(self, a: AgentNode) -> float:
        if a.q not in self.A.accepting:
            raise ValueError("terminal_value called on nonaccepting state.")
        if a in self._terminal_cache:
            return self._terminal_cache[a]
        val = self.oracle.terminal_shifted_cvar(a.K, self.alpha)
        self._terminal_cache[a] = val
        return val

    # ---------- optimistic lower bound ----------

    def optimistic_cost_to_accept(self, a: AgentNode) -> float:
        if a.q in self.A.accepting:
            return 0.0
        if a in self._optimistic_lb_cache:
            return self._optimistic_lb_cache[a]

        kd = dict(a.K)
        start = (a.x, a.q)
        dist = {start: 0.0}
        heap = [(0.0, start)]

        while heap:
            d, (x, q) = heapq.heappop(heap)
            if d != dist[(x, q)]:
                continue
            if q in self.A.accepting:
                self._optimistic_lb_cache[a] = d
                return d

            pats = self.T.patterns[x]
            if len(pats) == 1:
                succ = pats[0]
            elif x in kd:
                succ = pats[kd[x]]
            else:
                s = set()
                for pat in pats:
                    s.update(pat)
                succ = frozenset(s)

            for y in succ:
                q2 = self.A.step(q, self.T.labels[y])
                nd = d + self.T.weights[(x, y)]
                key = (y, q2)
                if nd < dist.get(key, inf):
                    dist[key] = nd
                    heapq.heappush(heap, (nd, key))

        self._optimistic_lb_cache[a] = inf
        return inf

    def action_lower_bound(self, env: EnvNode) -> float:
        probs = self.branches(env)
        vals = []
        ps = []
        for ch, p in probs.items():
            vals.append(0.0 if ch.q in self.A.accepting else self.optimistic_cost_to_accept(ch))
            ps.append(p)
        return self.action_cost(env) + discrete_cvar(vals, ps, self.alpha)

    # ---------- horizon Bellman recursion ----------

    def value(self, a: AgentNode, h: int) -> float:
        if a.q in self.A.accepting:
            return self.terminal_value(a)
        if h <= 0:
            return inf

        key = (a, h)
        if key in self._value_cache:
            return self._value_cache[key]

        self.value_state_budgets += 1
        feasible = []
        for env in self.actions(a):
            children = self.branches(env)
            if children and all(self.can_accept_within(ch, h - 1) for ch in children):
                feasible.append(env)

        if not feasible:
            self._value_cache[key] = inf
            return inf

        if self.use_lower_bound_pruning:
            feasible.sort(key=lambda e: (self.action_lower_bound(e), e.target))
        else:
            feasible.sort(key=lambda e: e.target)

        best = inf
        best_env = None

        for env in feasible:
            if self.use_lower_bound_pruning:
                lb = self.action_lower_bound(env)
                if lb > best + 1e-12:
                    self.pruned_actions += 1
                    continue

            probs = self.branches(env)
            child_values = [self.value(ch, h - 1) for ch in probs]
            if any(v == inf for v in child_values):
                continue
            child_probs = [probs[ch] for ch in probs]
            q = self.action_cost(env) + discrete_cvar(
                child_values, child_probs, self.alpha
            )

            if q < best - 1e-12 or (
                isclose(q, best, rel_tol=0.0, abs_tol=1e-12) and
                (best_env is None or env.target < best_env.target)
            ):
                best = q
                best_env = env

        if best_env is None:
            self._value_cache[key] = inf
            return inf

        self._value_cache[key] = best
        self.policy[key] = best_env
        return best

    def solve(self) -> FactoredDynamicCVaRResult:
        if not self.can_accept_within(self.start, self.horizon):
            raise ValueError(
                f"No robust policy accepts within horizon H={self.horizon}."
            )

        shifted = self.value(self.start, self.horizon)
        if shifted == inf:
            raise RuntimeError("Factored horizon problem returned infinite value.")

        return FactoredDynamicCVaRResult(
            alpha=self.alpha,
            horizon=self.horizon,
            shifted_value=shifted,
            dynamic_regret_value=shifted - self.oracle_reference,
            oracle_reference=self.oracle_reference,
            policy=dict(self.policy),
            generated_agent_states=len(self.generated_agent_states),
            generated_observation_branches=self.generated_observation_branches,
            value_state_budgets=self.value_state_budgets,
            pruned_actions=self.pruned_actions,
            symbolic_oracle_nodes=self.oracle.symbolic_nodes,
            symbolic_oracle_collapses=self.oracle.collapsed_nodes,
            oracle_shortest_path_calls=self.oracle.oracle_calls,
            conceptual_world_upper_bound=self.prior.conceptual_world_upper_bound(),
        )
