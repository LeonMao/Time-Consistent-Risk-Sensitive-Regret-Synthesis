
from __future__ import annotations

from collections import defaultdict
from math import inf, isclose
import heapq
from typing import Dict, Mapping, Tuple

from stage0_solver import DFA, Knowledge, PKWTS
from factored_belief import MixtureProductPrior
from stage1_3_risk import discrete_cvar


class MonotoneBinaryOracleDiagram:
    """
    Exact symbolic distribution of the clairvoyant temporal-logic oracle cost
    for binary monotone topology variables.

    Each unknown state must have two patterns with one a subset of the other:
        closed_pattern ⊆ open_pattern.

    For a partial assignment and one product component:
      - optimistic completion chooses open whenever supported;
      - pessimistic completion chooses closed whenever supported.

    Since adding transitions cannot increase shortest accepting cost,
        J_opt <= J*(completion) <= J_pess.

    If J_opt == J_pess, every completion under that symbolic node has the same
    oracle cost, so the entire remaining subtree is collapsed exactly.

    This is an oracle decision diagram / Shannon decomposition specialized to
    monotone PK-WTS topology.
    """

    def __init__(self, T: PKWTS, dfa: DFA, prior: MixtureProductPrior):
        self.T = T
        self.A = dfa
        self.prior = prior
        self.variables = prior.variables

        self.closed_idx: Dict[str, int] = {}
        self.open_idx: Dict[str, int] = {}
        for x in self.variables:
            pats = T.patterns[x]
            if len(pats) != 2:
                raise ValueError(
                    "Symbolic oracle currently requires binary unknown states."
                )
            p0, p1 = set(pats[0]), set(pats[1])
            if p0.issubset(p1):
                self.closed_idx[x], self.open_idx[x] = 0, 1
            elif p1.issubset(p0):
                self.closed_idx[x], self.open_idx[x] = 1, 0
            else:
                raise ValueError(
                    f"Patterns at {x} are not monotone by set inclusion."
                )

        self.state_index = T.state_index
        self.oracle_calls = 0
        self.symbolic_nodes = 0
        self.collapsed_nodes = 0
        self._oracle_cache: Dict[Tuple[int, ...], float] = {}
        self._dist_cache: Dict[
            Tuple[int, Tuple[Tuple[str, int], ...]], Dict[float, float]
        ] = {}

        # Variable order: preserve T.states order. Benchmarks can order variables
        # by expected relevance to improve compression.
        self.variable_order = tuple(x for x in T.states if x in self.closed_idx)

        self.oracle_reference = self._compute_max_supported_oracle()

    def _canonical_partial(
        self, partial: Mapping[str, int]
    ) -> Tuple[Tuple[str, int], ...]:
        """Canonical ODD assignment key in the fixed topology-variable order."""
        return tuple((x, partial[x]) for x in self.variable_order if x in partial)

    def _completion_tuple(
        self,
        mode: int,
        partial: Mapping[str, int],
        optimistic: bool,
    ) -> Tuple[int, ...]:
        vals = []
        comp = self.prior.components[mode]
        for x in self.T.states:
            pats = self.T.patterns[x]
            if len(pats) == 1:
                vals.append(0)
                continue
            if x in partial:
                vals.append(partial[x])
                continue

            c = self.closed_idx[x]
            o = self.open_idx[x]
            pc = comp.marginals[x][c]
            po = comp.marginals[x][o]
            if optimistic:
                if po > 0:
                    vals.append(o)
                elif pc > 0:
                    vals.append(c)
                else:
                    raise RuntimeError("Variable has no support in component.")
            else:
                if pc > 0:
                    vals.append(c)
                elif po > 0:
                    vals.append(o)
                else:
                    raise RuntimeError("Variable has no support in component.")
        return tuple(vals)

    def _oracle_cost_tuple(self, assignment: Tuple[int, ...]) -> float:
        if assignment in self._oracle_cache:
            return self._oracle_cache[assignment]

        self.oracle_calls += 1
        q0 = self.A.step(self.A.q_init, self.T.labels[self.T.x0])
        start = (self.T.x0, q0)
        dist = {start: 0.0}
        heap = [(0.0, start)]

        while heap:
            d, (x, q) = heapq.heappop(heap)
            if d != dist[(x, q)]:
                continue
            if q in self.A.accepting:
                self._oracle_cache[assignment] = d
                return d

            pidx = assignment[self.state_index[x]]
            for y in self.T.patterns[x][pidx]:
                q2 = self.A.step(q, self.T.labels[y])
                nd = d + self.T.weights[(x, y)]
                key = (y, q2)
                if nd < dist.get(key, inf):
                    dist[key] = nd
                    heapq.heappush(heap, (nd, key))

        self._oracle_cache[assignment] = inf
        return inf

    def _component_possible(self, mode: int, partial: Mapping[str, int]) -> bool:
        comp = self.prior.components[mode]
        for x, k in partial.items():
            if x in comp.marginals and comp.marginals[x][k] <= 0:
                return False
        return True

    def _compute_max_supported_oracle(self) -> float:
        worst = -inf
        for mode in range(len(self.prior.components)):
            if self.prior.components[mode].weight <= 0:
                continue
            assignment = self._completion_tuple(mode, {}, optimistic=False)
            val = self._oracle_cost_tuple(assignment)
            worst = max(worst, val)
        if worst == inf:
            raise ValueError("Some supported pessimistic completion cannot satisfy the task.")
        return worst

    def _component_distribution_recursive(
        self,
        mode: int,
        K: Knowledge,
        partial_extra: Tuple[Tuple[str, int], ...],
    ) -> Dict[float, float]:
        """Return P_c(J*=j | K and partial_extra) as a sparse mass map."""
        partial = dict(K)
        partial.update(dict(partial_extra))
        key = (mode, self._canonical_partial(partial))
        if key in self._dist_cache:
            return self._dist_cache[key]

        self.symbolic_nodes += 1

        if not self._component_possible(mode, partial):
            self._dist_cache[key] = {}
            return self._dist_cache[key]

        opt_assignment = self._completion_tuple(mode, partial, optimistic=True)
        pess_assignment = self._completion_tuple(mode, partial, optimistic=False)
        j_opt = self._oracle_cost_tuple(opt_assignment)
        j_pess = self._oracle_cost_tuple(pess_assignment)

        if isclose(j_opt, j_pess, rel_tol=0.0, abs_tol=1e-12):
            self.collapsed_nodes += 1
            out = {j_opt: 1.0}
            self._dist_cache[key] = out
            return out

        comp = self.prior.components[mode]
        branch_var = None
        for x in self.variable_order:
            if x in partial:
                continue
            c = self.closed_idx[x]
            o = self.open_idx[x]
            if comp.marginals[x][c] > 0 and comp.marginals[x][o] > 0:
                branch_var = x
                break

        if branch_var is None:
            # Full supported assignment (or all remaining variables deterministic).
            full = self._completion_tuple(mode, partial, optimistic=True)
            j = self._oracle_cost_tuple(full)
            out = {j: 1.0}
            self._dist_cache[key] = out
            return out

        c = self.closed_idx[branch_var]
        o = self.open_idx[branch_var]
        probs = comp.marginals[branch_var]
        support = [(k, probs[k]) for k in (c, o) if probs[k] > 0]
        z = sum(p for _, p in support)

        agg = defaultdict(float)
        extra_dict = dict(partial_extra)
        for k, p in support:
            next_extra = dict(extra_dict)
            next_extra[branch_var] = k
            sub = self._component_distribution_recursive(
                mode, K, self._canonical_partial(next_extra)
            )
            for j, mass in sub.items():
                agg[j] += (p / z) * mass

        out = dict(agg)
        self._dist_cache[key] = out
        return out

    def oracle_cost_distribution(self, K: Knowledge) -> Dict[float, float]:
        """Return the exact mixture posterior P(J*=j | K)."""
        mode_post = self.prior.component_posterior_weights(K)
        agg = defaultdict(float)

        for mode, w in enumerate(mode_post):
            if w <= 0:
                continue
            dist = self._component_distribution_recursive(mode, K, tuple())
            for j, p in dist.items():
                agg[j] += w * p

        z = sum(agg.values())
        if z <= 0:
            raise RuntimeError("Symbolic oracle distribution has zero mass.")
        return {j: p / z for j, p in agg.items()}

    def terminal_shifted_cvar(self, K: Knowledge, alpha: float) -> float:
        dist = self.oracle_cost_distribution(K)
        values = [self.oracle_reference - j for j in dist]
        probs = [dist[j] for j in dist]
        return discrete_cvar(values, probs, alpha)
