
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isclose
import heapq
from typing import Dict, FrozenSet, Mapping, Optional, Sequence, Set, Tuple

from stage0_solver import AgentNode, DFA, EnvNode, Knowledge, PKWTS, Stage0RegretSolver, World
from stage1_common import normalize_prior, restrict_to_prior_support
from stage1_3_risk import discrete_cvar


@dataclass
class LazyDynamicCVaRResult:
    alpha: float
    policy: Dict[AgentNode, EnvNode]
    shifted_value: float
    dynamic_regret_value: float
    oracle_reference: float
    start_rank: int
    generated_agent_states: int
    generated_env_actions: int
    generated_action_branches: int
    value_expanded_states: int
    pruned_actions_by_bound: int
    rank_queries: int
    policy_world_costs: Dict[World, float]
    policy_world_regrets: Dict[World, float]


class LazyProgressDynamicCVaRSolver:
    """
    Stage 1.5 implicit/lazy solver for the Stage-1.4 dynamic CVaR-regret model.

    The solver does NOT prebuild the complete knowledge game. Instead it:
      1) generates observation branches on demand from (x,q,K);
      2) finds a robust progress rank by depth-limited AND/OR recursion;
      3) retains only actions whose every branch can accept within rank-1;
      4) solves the resulting acyclic rank-decreasing dynamic-CVaR problem
         by memoized backward recursion;
      5) uses an admissible optimistic lower bound to prune actions.

    This is exact over the progress-policy class:
        Pi_prog = {policies whose selected actions strictly decrease robust rank}.

    If a globally optimal Stage-1.4 policy is progress-rank decreasing, this
    lazy solution equals the unrestricted deterministic reference solution.
    """

    def __init__(
        self,
        T: PKWTS,
        dfa: DFA,
        prior: Mapping[World, float],
        alpha: float,
        max_rank: Optional[int] = None,
        use_lower_bound_pruning: bool = True,
    ):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha < 1.")

        full_prior = normalize_prior(T, prior)
        self.prior = full_prior
        self.T = restrict_to_prior_support(T, full_prior)
        self.A = dfa
        self.alpha = float(alpha)
        self.worlds = tuple(full_prior.keys())
        self.max_rank = max_rank if max_rank is not None else max(
            4, 2 * len(self.T.states) * len(self.A.states)
        )
        self.use_lower_bound_pruning = bool(use_lower_bound_pruning)

        q0 = self.A.step(self.A.q_init, self.T.labels[self.T.x0])
        self.start = AgentNode(self.T.x0, q0, frozenset())

        # Oracle computation uses the frozen Stage-0 world-product shortest path
        # but never builds the Stage-0 knowledge game.
        self.oracle_engine = Stage0RegretSolver(self.T, self.A)
        self.oracle_costs = {w: self.oracle_engine.oracle_cost(w) for w in self.worlds}
        if any(v == inf for v in self.oracle_costs.values()):
            raise ValueError("At least one supported world has infinite clairvoyant task cost.")

        self.oracle_reference = max(self.oracle_costs.values())
        self.oracle_gap = {
            w: self.oracle_reference - self.oracle_costs[w]
            for w in self.worlds
        }

        # Caches / counters.
        self._posterior_cache: Dict[Knowledge, Dict[World, float]] = {}
        self._actions_cache: Dict[AgentNode, Tuple[EnvNode, ...]] = {}
        self._branches_cache: Dict[EnvNode, Dict[AgentNode, float]] = {}
        self._rank_feasible_cache: Dict[Tuple[AgentNode, int], bool] = {}
        self._min_rank_cache: Dict[AgentNode, int] = {}
        self._terminal_cache: Dict[AgentNode, float] = {}
        self._value_cache: Dict[AgentNode, float] = {}
        self._optimistic_lb_cache: Dict[AgentNode, float] = {}

        self.generated_agent_states: Set[AgentNode] = {self.start}
        self.generated_action_branches = 0
        self.value_expanded_states = 0
        self.pruned_actions_by_bound = 0
        self.rank_queries = 0
        self.policy: Dict[AgentNode, EnvNode] = {}

    # ---------- fixed-world Bayesian knowledge mechanics ----------

    def posterior(self, K: Knowledge) -> Dict[World, float]:
        if K in self._posterior_cache:
            return self._posterior_cache[K]

        kd = dict(K)
        masses = {}
        for w, p in self.prior.items():
            ok = True
            for x, pidx in kd.items():
                if self.T.world_pattern_index(w, x) != pidx:
                    ok = False
                    break
            if ok:
                masses[w] = p

        z = sum(masses.values())
        if z <= 0:
            raise RuntimeError("Knowledge is inconsistent with prior support.")

        post = {w: p / z for w, p in masses.items()}
        self._posterior_cache[K] = post
        return post

    def current_successors(self, a: AgentNode):
        pats = self.T.patterns[a.x]
        if len(pats) == 1:
            return pats[0]
        kd = dict(a.K)
        if a.x not in kd:
            raise RuntimeError(
                f"Unknown current state {a.x} has no revealed pattern in K."
            )
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

        post = self.posterior(env.K)
        probs: Dict[AgentNode, float] = {}

        for w, p in post.items():
            pidx = self.T.world_pattern_index(w, env.target)
            K2 = env.K
            if len(self.T.patterns[env.target]) > 1:
                kd = dict(K2)
                if env.target in kd and kd[env.target] != pidx:
                    continue
                K2 = frozenset(set(K2) | {(env.target, pidx)})

            q2 = self.A.step(env.q, self.T.labels[env.target])
            child = AgentNode(env.target, q2, K2)
            probs[child] = probs.get(child, 0.0) + p

        total = sum(probs.values())
        if total <= 0:
            raise RuntimeError("Action has no posterior-consistent observation branch.")
        probs = {z: p / total for z, p in probs.items()}

        self._branches_cache[env] = probs
        self.generated_action_branches += len(probs)
        self.generated_agent_states.update(probs)
        return probs

    def action_cost(self, env: EnvNode) -> float:
        return float(self.T.weights[(env.x, env.target)])

    # ---------- robust progress rank ----------

    def can_accept_within(self, a: AgentNode, depth: int) -> bool:
        """AND/OR robust reachability within at most `depth` physical moves."""
        self.rank_queries += 1
        key = (a, depth)
        if key in self._rank_feasible_cache:
            return self._rank_feasible_cache[key]

        if a.q in self.A.accepting:
            self._rank_feasible_cache[key] = True
            return True
        if depth <= 0:
            self._rank_feasible_cache[key] = False
            return False

        # Cheap actions first. This ordering does not change reachability, but
        # often finds a feasibility witness before expensive irrelevant branches
        # need to be generated.
        ordered_actions = sorted(
            self.actions(a), key=lambda e: (self.prebranch_action_lower_bound(e), e.target)
        )
        for env in ordered_actions:
            children = self.branches(env)
            if children and all(
                self.can_accept_within(ch, depth - 1)
                for ch in children
            ):
                self._rank_feasible_cache[key] = True
                return True

        self._rank_feasible_cache[key] = False
        return False

    def min_rank(self, a: AgentNode, upper: Optional[int] = None) -> int:
        if a.q in self.A.accepting:
            return 0
        if a in self._min_rank_cache:
            r = self._min_rank_cache[a]
            if upper is None or r <= upper:
                return r

        cap = self.max_rank if upper is None else min(self.max_rank, upper)
        for d in range(1, cap + 1):
            if self.can_accept_within(a, d):
                self._min_rank_cache[a] = d
                return d
        return inf  # type: ignore[return-value]

    def progress_actions(self, a: AgentNode, rank: int) -> Tuple[EnvNode, ...]:
        if rank <= 0:
            return tuple()
        out = []
        for env in self.actions(a):
            children = self.branches(env)
            if children and all(
                self.can_accept_within(ch, rank - 1)
                for ch in children
            ):
                out.append(env)
        return tuple(out)

    # ---------- shifted terminal/oracle risk ----------

    def terminal_value(self, a: AgentNode) -> float:
        if a.q not in self.A.accepting:
            raise ValueError("terminal_value called on nonaccepting state.")
        if a in self._terminal_cache:
            return self._terminal_cache[a]

        post = self.posterior(a.K)
        worlds = list(post)
        val = discrete_cvar(
            [self.oracle_gap[w] for w in worlds],
            [post[w] for w in worlds],
            self.alpha,
        )
        self._terminal_cache[a] = val
        return val

    # ---------- admissible optimistic lower bound ----------

    def optimistic_cost_to_accept(self, a: AgentNode) -> float:
        """
        Lower bound on future shifted risk value:
        shortest accepting cost in the optimistic refined skeleton, ignoring
        terminal oracle gap (which is nonnegative).
        """
        if a.q in self.A.accepting:
            return 0.0
        if a in self._optimistic_lb_cache:
            return self._optimistic_lb_cache[a]

        start = (a.x, a.q)
        dist = {start: 0.0}
        heap = [(0.0, start)]

        kd = dict(a.K)

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
                nd = d + float(self.T.weights[(x, y)])
                key = (y, q2)
                if nd < dist.get(key, inf):
                    dist[key] = nd
                    heapq.heappush(heap, (nd, key))

        self._optimistic_lb_cache[a] = inf
        return inf

    def prebranch_action_lower_bound(self, env: EnvNode) -> float:
        """Admissible action lower bound available before observation expansion.

        Uses the physical edge cost plus an optimistic refined-skeleton distance
        from the target. If the target is still unknown, its union successor
        pattern is used. Terminal oracle gap is nonnegative and omitted.
        """
        q2 = self.A.step(env.q, self.T.labels[env.target])
        pseudo = AgentNode(env.target, q2, env.K)
        return self.action_cost(env) + self.optimistic_cost_to_accept(pseudo)

    def action_lower_bound(self, env: EnvNode) -> float:
        probs = self.branches(env)
        child_lbs = []
        child_probs = []
        for child, p in probs.items():
            if child.q in self.A.accepting:
                lb = 0.0
            else:
                lb = self.optimistic_cost_to_accept(child)
            child_lbs.append(lb)
            child_probs.append(p)
        return self.action_cost(env) + discrete_cvar(
            child_lbs, child_probs, self.alpha
        )

    # ---------- lazy backward dynamic-CVaR recursion ----------

    def value(self, a: AgentNode) -> float:
        if a.q in self.A.accepting:
            return self.terminal_value(a)
        if a in self._value_cache:
            return self._value_cache[a]

        self.value_expanded_states += 1
        rank = self.min_rank(a)
        if rank == inf:
            raise ValueError(
                "No robust progress policy found within max_rank; "
                "increase max_rank or the task is not robustly realizable."
            )

        actions = list(self.progress_actions(a, rank))
        if not actions:
            raise RuntimeError("Finite-rank state has no progress action.")

        if self.use_lower_bound_pruning:
            actions.sort(key=lambda e: (self.action_lower_bound(e), e.target))
        else:
            actions.sort(key=lambda e: e.target)

        best = inf
        best_env = None

        for env in actions:
            if self.use_lower_bound_pruning:
                lb = self.action_lower_bound(env)
                if lb > best + 1e-12:
                    self.pruned_actions_by_bound += 1
                    continue

            probs = self.branches(env)
            vals = [self.value(ch) for ch in probs]
            ps = [probs[ch] for ch in probs]
            q = self.action_cost(env) + discrete_cvar(vals, ps, self.alpha)

            if q < best - 1e-12 or (
                isclose(q, best, rel_tol=0.0, abs_tol=1e-12) and
                (best_env is None or env.target < best_env.target)
            ):
                best = q
                best_env = env

        if best_env is None:
            # Can happen only if first action was pruned against +inf, which is impossible.
            raise RuntimeError("No action evaluated.")

        self._value_cache[a] = best
        self.policy[a] = best_env
        return best

    # ---------- fixed-world execution for validation ----------

    def simulate_policy_world(self, world: World, max_steps: Optional[int] = None):
        if max_steps is None:
            max_steps = self.max_rank + 2
        a = self.start
        cost = 0.0

        for _ in range(max_steps):
            if a.q in self.A.accepting:
                return cost, a
            if a not in self.policy:
                # Ensure a reachable selected branch has been solved.
                self.value(a)
            env = self.policy[a]
            pidx = self.T.world_pattern_index(world, env.target)
            K2 = env.K
            if len(self.T.patterns[env.target]) > 1:
                K2 = frozenset(set(K2) | {(env.target, pidx)})
            q2 = self.A.step(env.q, self.T.labels[env.target])
            a = AgentNode(env.target, q2, K2)
            cost += self.action_cost(env)

        raise RuntimeError("Progress policy exceeded its finite rank bound.")

    def solve(self) -> LazyDynamicCVaRResult:
        start_rank = self.min_rank(self.start)
        if start_rank == inf:
            raise ValueError(
                f"No robust progress policy found within max_rank={self.max_rank}."
            )

        shifted = self.value(self.start)
        dyn_regret = shifted - self.oracle_reference

        costs = {}
        regrets = {}
        for w in self.worlds:
            c, _ = self.simulate_policy_world(w, max_steps=start_rank + 1)
            costs[w] = c
            regrets[w] = c - self.oracle_costs[w]

        return LazyDynamicCVaRResult(
            alpha=self.alpha,
            policy=dict(self.policy),
            shifted_value=shifted,
            dynamic_regret_value=dyn_regret,
            oracle_reference=self.oracle_reference,
            start_rank=start_rank,
            generated_agent_states=len(self.generated_agent_states),
            generated_env_actions=sum(len(v) for v in self._actions_cache.values()),
            generated_action_branches=self.generated_action_branches,
            value_expanded_states=self.value_expanded_states,
            pruned_actions_by_bound=self.pruned_actions_by_bound,
            rank_queries=self.rank_queries,
            policy_world_costs=costs,
            policy_world_regrets=regrets,
        )


def explicit_robust_progress_ranks(game: Stage0RegretSolver):
    """Compute exact robust physical-step ranks on an already-built game.

    r(z)=0 for accepting agent states.
    r(z)=1+min_a max_{z' in Succ(z,a)} r(z') for finite-rank states.

    Returns:
      rank: AgentNode -> nonnegative integer
      progress_actions: AgentNode -> tuple[EnvNode,...] whose branches all
                        have rank strictly smaller than the parent rank.
    """
    rank: Dict[AgentNode, int] = {
        a: 0 for a in game.accepting_nodes
    }

    changed = True
    k = 0
    while changed:
        changed = False
        k += 1
        for a in game.agent_nodes:
            if a in rank:
                continue
            for env, _ in game.adj[a]:
                if not isinstance(env, EnvNode):
                    continue
                children = [z for z, _ in game.adj[env]]
                if children and all(
                    isinstance(ch, AgentNode) and ch in rank and rank[ch] <= k - 1
                    for ch in children
                ):
                    rank[a] = k
                    changed = True
                    break

    prog: Dict[AgentNode, Tuple[EnvNode, ...]] = {}
    for a, r in rank.items():
        if r == 0:
            prog[a] = tuple()
            continue
        out = []
        for env, _ in game.adj[a]:
            if not isinstance(env, EnvNode):
                continue
            children = [z for z, _ in game.adj[env]]
            if children and all(
                isinstance(ch, AgentNode) and ch in rank and rank[ch] < r
                for ch in children
            ):
                out.append(env)
        prog[a] = tuple(out)

    return rank, prog


@dataclass
class LazyHorizonDynamicCVaRResult:
    alpha: float
    horizon: int
    policy: Dict[Tuple[AgentNode, int], EnvNode]
    shifted_value: float
    dynamic_regret_value: float
    oracle_reference: float
    generated_agent_states: int
    generated_env_actions: int
    generated_action_branches: int
    value_expanded_state_budgets: int
    pruned_actions_by_bound: int
    policy_world_costs: Dict[World, float]
    policy_world_regrets: Dict[World, float]


class LazyHorizonDynamicCVaRSolver(LazyProgressDynamicCVaRSolver):
    """
    Exact implicit solver over Pi_H: deterministic contingent policies that
    robustly accept within at most H physical transitions.

    Unlike the minimal-rank solver, this class does not require every action to
    decrease the *minimal* robust rank. It augments the computation with a
    remaining-step certificate h. Thus exploratory detours are permitted as
    long as every observation branch can still accept within h-1 steps.

    For finite-world deterministic PK-WTSs, every deterministic robust proper
    policy has a finite uniform worst-case hitting time H_pi. Hence the classes

        Pi_0 subset Pi_1 subset ... subset Pi_H subset ...

    exhaust all deterministic robust proper policies. Therefore there exists a
    finite H* for any optimal proper policy, and for all H >= H* the budgeted
    solver recovers the unrestricted deterministic proper optimum.
    """

    def __init__(
        self,
        T: PKWTS,
        dfa: DFA,
        prior: Mapping[World, float],
        alpha: float,
        horizon: int,
        use_lower_bound_pruning: bool = True,
    ):
        if horizon < 0:
            raise ValueError("horizon must be nonnegative.")
        super().__init__(
            T=T,
            dfa=dfa,
            prior=prior,
            alpha=alpha,
            max_rank=horizon,
            use_lower_bound_pruning=use_lower_bound_pruning,
        )
        self.horizon = int(horizon)
        self._budget_value_cache: Dict[Tuple[AgentNode, int], float] = {}
        self.budget_policy: Dict[Tuple[AgentNode, int], EnvNode] = {}
        self.value_expanded_state_budgets = 0

    def budget_value(self, a: AgentNode, h: int) -> float:
        if a.q in self.A.accepting:
            return self.terminal_value(a)
        if h <= 0:
            return inf

        key = (a, h)
        if key in self._budget_value_cache:
            return self._budget_value_cache[key]

        self.value_expanded_state_budgets += 1

        # Process actions lazily. Immediate physical cost is already an
        # admissible lower bound because all shifted child/terminal values are
        # nonnegative. Therefore, once an incumbent is available, an expensive
        # action can be pruned *before* generating its observation branches.
        ordered = sorted(
            self.actions(a), key=lambda e: (self.prebranch_action_lower_bound(e), e.target)
        )

        best = inf
        best_env = None

        for env in ordered:
            if self.use_lower_bound_pruning and self.prebranch_action_lower_bound(env) > best + 1e-12:
                self.pruned_actions_by_bound += 1
                continue

            probs = self.branches(env)
            if not probs or not all(
                self.can_accept_within(ch, h - 1) for ch in probs
            ):
                continue

            if self.use_lower_bound_pruning:
                lb = self.action_lower_bound(env)
                if lb > best + 1e-12:
                    self.pruned_actions_by_bound += 1
                    continue

            vals = [self.budget_value(ch, h - 1) for ch in probs]
            if any(v == inf for v in vals):
                continue
            ps = [probs[ch] for ch in probs]
            q = self.action_cost(env) + discrete_cvar(vals, ps, self.alpha)

            if q < best - 1e-12 or (
                isclose(q, best, rel_tol=0.0, abs_tol=1e-12) and
                (best_env is None or env.target < best_env.target)
            ):
                best = q
                best_env = env

        if best_env is None:
            self._budget_value_cache[key] = inf
            return inf

        self._budget_value_cache[key] = best
        self.budget_policy[key] = best_env
        return best

    def simulate_budget_policy_world(self, world: World):
        a = self.start
        h = self.horizon
        cost = 0.0

        while True:
            if a.q in self.A.accepting:
                return cost, a
            if h <= 0:
                raise RuntimeError("Budget policy failed to accept within H.")

            key = (a, h)
            if key not in self.budget_policy:
                self.budget_value(a, h)
            env = self.budget_policy[key]

            pidx = self.T.world_pattern_index(world, env.target)
            K2 = env.K
            if len(self.T.patterns[env.target]) > 1:
                K2 = frozenset(set(K2) | {(env.target, pidx)})
            q2 = self.A.step(env.q, self.T.labels[env.target])
            a = AgentNode(env.target, q2, K2)
            cost += self.action_cost(env)
            h -= 1

    def solve(self) -> LazyHorizonDynamicCVaRResult:
        if not self.can_accept_within(self.start, self.horizon):
            raise ValueError(
                f"No robust policy can accept within horizon H={self.horizon}."
            )

        shifted = self.budget_value(self.start, self.horizon)
        if shifted == inf:
            raise RuntimeError("Feasible finite-horizon problem returned infinite value.")

        dyn_regret = shifted - self.oracle_reference

        costs = {}
        regrets = {}
        for w in self.worlds:
            c, _ = self.simulate_budget_policy_world(w)
            costs[w] = c
            regrets[w] = c - self.oracle_costs[w]

        return LazyHorizonDynamicCVaRResult(
            alpha=self.alpha,
            horizon=self.horizon,
            policy=dict(self.budget_policy),
            shifted_value=shifted,
            dynamic_regret_value=dyn_regret,
            oracle_reference=self.oracle_reference,
            generated_agent_states=len(self.generated_agent_states),
            generated_env_actions=sum(len(v) for v in self._actions_cache.values()),
            generated_action_branches=self.generated_action_branches,
            value_expanded_state_budgets=self.value_expanded_state_budgets,
            pruned_actions_by_bound=self.pruned_actions_by_bound,
            policy_world_costs=costs,
            policy_world_regrets=regrets,
        )
