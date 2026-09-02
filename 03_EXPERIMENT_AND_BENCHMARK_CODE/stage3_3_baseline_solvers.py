
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isclose
from typing import Dict, Mapping, Tuple

from stage0_solver import AgentNode, EnvNode
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from stage1_3_risk import discrete_cvar


@dataclass
class BaselineResult:
    policy: Dict[Tuple[AgentNode, int], EnvNode]
    objective: float


class FactoredDynamicCVaRCostSolver(FactoredLazyHorizonDynamicCVaRSolver):
    """Nested-CVaR planner for absolute physical mission cost, not regret."""

    def terminal_value(self, a: AgentNode) -> float:
        if a.q not in self.A.accepting:
            raise ValueError("terminal_value called on nonaccepting state")
        return 0.0

    def solve_cost(self) -> BaselineResult:
        if not self.can_accept_within(self.start, self.horizon):
            raise ValueError("No H-proper robust policy.")
        v = self.value(self.start, self.horizon)
        return BaselineResult(dict(self.policy), float(v))


class FactoredWorstCaseCostSolver(FactoredLazyHorizonDynamicCVaRSolver):
    """Minimize worst-case absolute physical mission cost over H-proper policies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wc_cache = {}
        self.wc_policy = {}

    def wc_value(self, a: AgentNode, h: int) -> float:
        if a.q in self.A.accepting:
            return 0.0
        if h <= 0:
            return inf
        key = (a,h)
        if key in self._wc_cache:
            return self._wc_cache[key]

        best = inf
        best_env = None
        for env in self.actions(a):
            children = self.branches(env)
            if not children or not all(self.can_accept_within(ch,h-1) for ch in children):
                continue
            vals = [self.wc_value(ch,h-1) for ch in children]
            if any(v == inf for v in vals):
                continue
            q = self.action_cost(env) + max(vals)
            if q < best - 1e-12 or (
                isclose(q,best,abs_tol=1e-12) and
                (best_env is None or repr(env) < repr(best_env))
            ):
                best=q; best_env=env

        self._wc_cache[key]=best
        if best_env is not None:
            self.wc_policy[key]=best_env
        return best

    def solve_cost(self) -> BaselineResult:
        if not self.can_accept_within(self.start,self.horizon):
            raise ValueError("No H-proper robust policy.")
        v=self.wc_value(self.start,self.horizon)
        return BaselineResult(dict(self.wc_policy),float(v))


class FactoredWorstCaseRegretSolver(FactoredLazyHorizonDynamicCVaRSolver):
    """H-bounded minimax hindsight regret (Zhao-style objective)."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._mr_cache={}
        self.mr_policy={}
        self._terminal_max_cache={}

    def terminal_max_gap(self,a:AgentNode)->float:
        if a in self._terminal_max_cache:
            return self._terminal_max_cache[a]
        dist=self.oracle.oracle_cost_distribution(a.K)
        val=max(self.oracle_reference-j for j,p in dist.items() if p>0)
        self._terminal_max_cache[a]=val
        return val

    def mr_shifted_value(self,a:AgentNode,h:int)->float:
        if a.q in self.A.accepting:
            return self.terminal_max_gap(a)
        if h<=0:
            return inf
        key=(a,h)
        if key in self._mr_cache:
            return self._mr_cache[key]
        best=inf;best_env=None
        for env in self.actions(a):
            children=self.branches(env)
            if not children or not all(self.can_accept_within(ch,h-1) for ch in children):
                continue
            vals=[self.mr_shifted_value(ch,h-1) for ch in children]
            if any(v==inf for v in vals):
                continue
            q=self.action_cost(env)+max(vals)
            if q<best-1e-12 or (
                isclose(q,best,abs_tol=1e-12) and
                (best_env is None or repr(env)<repr(best_env))
            ):
                best=q;best_env=env
        self._mr_cache[key]=best
        if best_env is not None:
            self.mr_policy[key]=best_env
        return best

    def solve_regret(self)->BaselineResult:
        if not self.can_accept_within(self.start,self.horizon):
            raise ValueError("No H-proper robust policy.")
        shifted=self.mr_shifted_value(self.start,self.horizon)
        return BaselineResult(dict(self.mr_policy),float(shifted-self.oracle_reference))
