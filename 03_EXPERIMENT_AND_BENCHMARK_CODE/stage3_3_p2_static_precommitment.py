
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import inf
from typing import Dict, Mapping, Tuple

from stage0_solver import AgentNode, EnvNode, PKWTS, DFA, Stage0RegretSolver, World
from stage1_3_risk import discrete_cvar
from stage1_common import normalize_prior


@dataclass
class StaticPolicyCandidate:
    future_costs: Dict[World, float]
    policy: Dict[Tuple[AgentNode, int], EnvNode]


@dataclass
class StaticPrecommitmentResult:
    alpha: float
    horizon: int
    cvar_regret: float
    expected_regret: float
    worst_regret: float
    world_costs: Dict[World, float]
    world_regrets: Dict[World, float]
    policy: Dict[Tuple[AgentNode, int], EnvNode]
    candidates_considered: int


class ExactStaticCVaRPrecommitmentSolver:
    """
    Exact finite-H static/precommitment CVaR-regret synthesis for small
    explicit-world PK-WTSs.

    It enumerates deterministic H-proper contingent policy trees by recursively
    combining one continuation per posterior observation branch.

    This is deliberately a small-instance ground-truth solver for P2-2, not a
    scalable algorithm.
    """

    def __init__(
        self,
        T: PKWTS,
        dfa: DFA,
        prior: Mapping[World, float],
        alpha: float,
        horizon: int,
    ):
        if not 0 <= alpha < 1:
            raise ValueError("alpha must be in [0,1).")
        self.T = T
        self.A = dfa
        self.prior = normalize_prior(T, prior)
        self.alpha = float(alpha)
        self.horizon = int(horizon)
        self.game = Stage0RegretSolver(T, dfa)
        self.worlds = tuple(self.prior.keys())
        self.oracle = {w:self.game.oracle_cost(w) for w in self.worlds}
        if any(v == inf for v in self.oracle.values()):
            raise ValueError("A supported world has infinite clairvoyant oracle cost.")

        q0 = self.A.step(self.A.q_init, self.T.labels[self.T.x0])
        self.start = AgentNode(self.T.x0, q0, frozenset())
        self._enum_cache = {}

    def compatible_worlds(self, K):
        kd = dict(K)
        return tuple(
            w for w in self.worlds
            if all(self.T.world_pattern_index(w,x) == p for x,p in kd.items())
        )

    def current_successors(self, a: AgentNode):
        pats = self.T.patterns[a.x]
        if len(pats) == 1:
            return pats[0]
        kd = dict(a.K)
        if a.x not in kd:
            raise RuntimeError(f"unknown current state {a.x} not observed")
        return pats[kd[a.x]]

    def actions(self, a: AgentNode):
        if a.q in self.A.accepting:
            return tuple()
        return tuple(EnvNode(a.x,a.q,a.K,y) for y in sorted(self.current_successors(a)))

    def child_for_world(self, env: EnvNode, w: World):
        pidx = self.T.world_pattern_index(w, env.target)
        K2 = env.K
        if len(self.T.patterns[env.target]) > 1:
            K2 = frozenset(set(K2) | {(env.target,pidx)})
        q2 = self.A.step(env.q, self.T.labels[env.target])
        return AgentNode(env.target,q2,K2)

    def enumerate_from(self, a: AgentNode, h: int):
        key=(a,h)
        if key in self._enum_cache:
            return self._enum_cache[key]

        worlds = self.compatible_worlds(a.K)

        if a.q in self.A.accepting:
            out=(StaticPolicyCandidate({w:0.0 for w in worlds}, {}),)
            self._enum_cache[key]=out
            return out

        if h <= 0:
            self._enum_cache[key]=tuple()
            return tuple()

        candidates=[]
        seen_vectors=set()

        for env in self.actions(a):
            # Partition compatible worlds by posterior child.
            branch_worlds={}
            for w in worlds:
                ch=self.child_for_world(env,w)
                branch_worlds.setdefault(ch,[]).append(w)

            child_candidate_lists=[]
            feasible=True
            for ch in branch_worlds:
                cc=self.enumerate_from(ch,h-1)
                if not cc:
                    feasible=False
                    break
                child_candidate_lists.append((ch,cc))
            if not feasible:
                continue

            for combo in product(*[cc for _,cc in child_candidate_lists]):
                costs={w:float(self.T.weights[(env.x,env.target)]) for w in worlds}
                pol={(a,h):env}
                valid=True
                for ((ch,_), cand) in zip(child_candidate_lists,combo):
                    # Merge continuation policy and costs only on worlds in this child.
                    for k,v in cand.policy.items():
                        if k in pol and pol[k] != v:
                            valid=False
                            break
                        pol[k]=v
                    if not valid:
                        break
                    for w,c in cand.future_costs.items():
                        costs[w]+=c
                if not valid:
                    continue

                # Deduplicate identical world-cost vectors; static objective only
                # depends on the vector, deterministic tie-breaking picks repr policy.
                sig=tuple(round(costs[w],12) for w in worlds)
                if sig in seen_vectors:
                    continue
                seen_vectors.add(sig)
                candidates.append(StaticPolicyCandidate(costs,pol))

        candidates.sort(
            key=lambda c: (
                tuple(c.future_costs[w] for w in worlds),
                repr(sorted((repr(k),repr(v)) for k,v in c.policy.items()))
            )
        )
        out=tuple(candidates)
        self._enum_cache[key]=out
        return out

    def evaluate_candidate(
        self,
        a: AgentNode,
        candidate: StaticPolicyCandidate,
        past_cost: float = 0.0,
    ):
        worlds=self.compatible_worlds(a.K)
        mass=sum(self.prior[w] for w in worlds)
        probs=[self.prior[w]/mass for w in worlds]
        regrets=[
            past_cost + candidate.future_costs[w] - self.oracle[w]
            for w in worlds
        ]
        cv=discrete_cvar(regrets,probs,self.alpha)
        er=sum(p*r for p,r in zip(probs,regrets))
        wr=max(regrets)
        return cv,er,wr,dict(zip(worlds,regrets))

    def solve_from(
        self,
        a: AgentNode,
        h: int,
        past_cost: float = 0.0,
    ) -> StaticPrecommitmentResult:
        candidates=self.enumerate_from(a,h)
        if not candidates:
            raise ValueError("No H-proper contingent policy from this state.")

        ranked=[]
        evals=[]
        for i,c in enumerate(candidates):
            cv,er,wr,regs=self.evaluate_candidate(a,c,past_cost)
            ranked.append((cv,er,wr,i))
            evals.append((c,regs))
        ranked.sort()
        cv,er,wr,i=ranked[0]
        c,regs=evals[i]

        return StaticPrecommitmentResult(
            alpha=self.alpha,
            horizon=h,
            cvar_regret=cv,
            expected_regret=er,
            worst_regret=wr,
            world_costs={w:past_cost+c.future_costs[w] for w in c.future_costs},
            world_regrets=regs,
            policy=dict(c.policy),
            candidates_considered=len(candidates),
        )

    def solve(self):
        return self.solve_from(self.start,self.horizon,0.0)
