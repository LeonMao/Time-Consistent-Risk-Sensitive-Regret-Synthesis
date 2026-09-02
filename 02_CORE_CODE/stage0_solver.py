
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from math import inf, isinf
import heapq
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

State = str
DFAState = str
Pattern = FrozenSet[State]
Knowledge = FrozenSet[Tuple[State, int]]
World = Tuple[int, ...]

@dataclass(frozen=True)
class DFA:
    states: Tuple[DFAState, ...]
    q_init: DFAState
    accepting: FrozenSet[DFAState]
    transition_fn: Callable[[DFAState, FrozenSet[str]], DFAState]
    def step(self, q, label):
        q2 = self.transition_fn(q, label)
        if q2 not in self.states:
            raise ValueError(f"Unknown DFA state {q2}")
        return q2

@dataclass(frozen=True)
class PKWTS:
    states: Tuple[State, ...]
    x0: State
    patterns: Mapping[State, Tuple[Pattern, ...]]
    weights: Mapping[Tuple[State, State], float]
    labels: Mapping[State, FrozenSet[str]]
    allowed_worlds: Optional[Tuple[World, ...]] = None
    def __post_init__(self):
        ss = set(self.states)
        if self.x0 not in ss: raise ValueError("x0 not in states")
        for x in self.states:
            if x not in self.patterns or not self.patterns[x]:
                raise ValueError(f"Missing patterns for {x}")
            if x not in self.labels:
                raise ValueError(f"Missing label for {x}")
            for pat in self.patterns[x]:
                if not set(pat).issubset(ss):
                    raise ValueError(f"Bad successor in pattern of {x}")
                for y in pat:
                    if (x,y) not in self.weights:
                        raise ValueError(f"Missing weight {(x,y)}")
                    if self.weights[(x,y)] <= 0:
                        raise ValueError("Weights must be positive")
        if len(self.patterns[self.x0]) != 1:
            raise ValueError("Stage-0 solver currently requires x0 to be known")
    @property
    def state_index(self):
        return {x:i for i,x in enumerate(self.states)}
    def all_worlds(self):
        if self.allowed_worlds is not None:
            return self.allowed_worlds
        return tuple(tuple(v) for v in product(*[range(len(self.patterns[x])) for x in self.states]))
    def world_pattern_index(self, world, x):
        return world[self.state_index[x]]
    def world_successors(self, world, x):
        return self.patterns[x][self.world_pattern_index(world,x)]
    def skeleton_successors(self, x):
        s=set()
        for p in self.patterns[x]: s.update(p)
        return frozenset(s)

@dataclass(frozen=True)
class AgentNode:
    x: State
    q: DFAState
    K: Knowledge

@dataclass(frozen=True)
class EnvNode:
    x: State
    q: DFAState
    K: Knowledge
    target: State

GameNode = AgentNode | EnvNode

@dataclass
class Stage0Result:
    value: float
    policy: Dict[AgentNode, EnvNode]
    V: Dict[GameNode, float]
    mu: Dict[Tuple[GameNode,GameNode], float]
    dist: Dict[GameNode, float]
    shortest_path_edges: FrozenSet[Tuple[GameNode,GameNode]]
    oracle_costs: Dict[World,float]

class Stage0RegretSolver:
    def __init__(self, T:PKWTS, A:DFA):
        self.T, self.A = T, A
        self.worlds = T.all_worlds()
        self._oracle={}
        self._br={}
        self.adj={}
        self.pred={}
        q0=A.step(A.q_init,T.labels[T.x0])
        self.start=AgentNode(T.x0,q0,frozenset())

    def compatible_worlds(self,K):
        kd=dict(K); out=[]
        for w in self.worlds:
            if all(self.T.world_pattern_index(w,x)==p for x,p in kd.items()):
                out.append(w)
        return tuple(out)

    def current_successors(self,x,K):
        pats=self.T.patterns[x]
        if len(pats)==1: return pats[0]
        kd=dict(K)
        if x not in kd: raise RuntimeError(f"Unknown current state {x} not observed")
        return pats[kd[x]]

    def observation_indices(self,target,K):
        pats=self.T.patterns[target]
        if len(pats)==1: return (0,)
        kd=dict(K)
        if target in kd: return (kd[target],)
        return tuple(sorted({self.T.world_pattern_index(w,target) for w in self.compatible_worlds(K)}))

    def update_knowledge(self,K,target,pidx):
        if len(self.T.patterns[target])==1: return K
        kd=dict(K)
        if target in kd and kd[target]!=pidx: raise RuntimeError("Inconsistent observation")
        return frozenset(set(K)|{(target,pidx)})

    def oracle_cost(self,world):
        if world in self._oracle: return self._oracle[world]
        start=(self.T.x0,self.start.q)
        dist={start:0.0}; heap=[(0.0,start)]
        while heap:
            d,(x,q)=heapq.heappop(heap)
            if d!=dist[(x,q)]: continue
            if q in self.A.accepting:
                self._oracle[world]=d; return d
            for y in self.T.world_successors(world,x):
                q2=self.A.step(q,self.T.labels[y]); nd=d+self.T.weights[(x,y)]
                if nd<dist.get((y,q2),inf):
                    dist[(y,q2)]=nd; heapq.heappush(heap,(nd,(y,q2)))
        self._oracle[world]=inf; return inf

    def _add_edge(self,u,v,w):
        if (v,w) not in self.adj.setdefault(u,[]):
            self.adj[u].append((v,w)); self.pred.setdefault(v,[]).append((u,w))
        self.adj.setdefault(v,[]); self.pred.setdefault(u,[])

    def build_game(self):
        self.adj={self.start:[]}; self.pred={self.start:[]}
        queue=[self.start]; seen={self.start}
        while queue:
            n=queue.pop(0)
            if isinstance(n,AgentNode):
                if n.q in self.A.accepting: continue
                for y in sorted(self.current_successors(n.x,n.K)):
                    e=EnvNode(n.x,n.q,n.K,y); self._add_edge(n,e,0.0)
                    if e not in seen: seen.add(e); queue.append(e)
            else:
                for pidx in self.observation_indices(n.target,n.K):
                    K2=self.update_knowledge(n.K,n.target,pidx)
                    q2=self.A.step(n.q,self.T.labels[n.target])
                    a=AgentNode(n.target,q2,K2)
                    self._add_edge(n,a,self.T.weights[(n.x,n.target)])
                    if a not in seen: seen.add(a); queue.append(a)

    @property
    def agent_nodes(self): return tuple(n for n in self.adj if isinstance(n,AgentNode))
    @property
    def env_nodes(self): return tuple(n for n in self.adj if isinstance(n,EnvNode))
    @property
    def accepting_nodes(self): return tuple(n for n in self.agent_nodes if n.q in self.A.accepting)

    def best_response(self,K):
        if K in self._br: return self._br[K]
        ws=self.compatible_worlds(K)
        val=min((self.oracle_cost(w) for w in ws), default=inf)
        self._br[K]=val; return val

    def shortest_distances(self):
        dist={n:inf for n in self.adj}; dist[self.start]=0.0
        heap=[(0.0,0,self.start)]; seq=1
        while heap:
            d,_,u=heapq.heappop(heap)
            if d!=dist[u]: continue
            for v,w in self.adj[u]:
                nd=d+w
                if nd<dist[v]:
                    dist[v]=nd; heapq.heappush(heap,(nd,seq,v)); seq+=1
        return dist

    def shortest_path_edge_set(self,dist):
        spred={v:[] for v in self.adj}
        for u,edges in self.adj.items():
            for v,w in edges:
                if not isinf(dist[u]) and abs(dist[u]+w-dist[v])<=1e-12:
                    spred[v].append(u)
        marked=set(); stack=[f for f in self.accepting_nodes if not isinf(dist[f])]; seen=set(stack)
        while stack:
            v=stack.pop()
            for u in spred[v]:
                marked.add((u,v))
                if u not in seen: seen.add(u); stack.append(u)
        return frozenset(marked)

    def build_mu(self,dist,E_SP):
        mu={}
        for u,edges in self.adj.items():
            for v,_ in edges:
                if isinstance(u,AgentNode):
                    mu[(u,v)]=0.0
                elif (u,v) not in E_SP:
                    mu[(u,v)]=inf
                elif isinstance(v,AgentNode) and v.q in self.A.accepting:
                    mu[(u,v)]=dist[v]-self.best_response(v.K)
                else:
                    mu[(u,v)]=0.0
        return mu

    def minmax_value_iteration(self,mu,max_iter=None):
        if max_iter is None: max_iter=max(10,4*len(self.adj)+10)
        V={n:(0.0 if isinstance(n,AgentNode) and n.q in self.A.accepting else inf) for n in self.adj}
        def same(a,b): return (isinf(a) and isinf(b)) or abs(a-b)<=1e-12
        for _ in range(max_iter):
            NV=dict(V)
            for u,edges in self.adj.items():
                if isinstance(u,AgentNode) and u.q in self.A.accepting: NV[u]=0.0; continue
                if not edges: NV[u]=inf; continue
                vals=[mu[(u,v)]+V[v] for v,_ in edges]
                NV[u]=min(vals) if isinstance(u,AgentNode) else max(vals)
            if all(same(NV[n],V[n]) for n in V):
                V=NV; break
            V=NV
        else:
            raise RuntimeError("Value iteration did not converge")
        policy={}
        for u in self.agent_nodes:
            if u.q in self.A.accepting or not self.adj[u]: continue
            cand=sorted((mu[(u,v)]+V[v],repr(v),v) for v,_ in self.adj[u])
            if not isinf(cand[0][0]): policy[u]=cand[0][2]
        return V,policy

    def solve(self):
        self.build_game()
        oracle={w:self.oracle_cost(w) for w in self.worlds}
        dist=self.shortest_distances()
        E_SP=self.shortest_path_edge_set(dist)
        mu=self.build_mu(dist,E_SP)
        V,policy=self.minmax_value_iteration(mu)
        return Stage0Result(V[self.start],policy,V,mu,dist,E_SP,oracle)

    def simulate_policy(self,policy,world,max_steps=1000):
        node=self.start; cost=0.0
        for _ in range(max_steps):
            if node.q in self.A.accepting: return cost,node
            env=policy[node]; target=env.target
            pidx=self.T.world_pattern_index(world,target)
            K2=self.update_knowledge(node.K,target,pidx)
            q2=self.A.step(node.q,self.T.labels[target])
            nxt=AgentNode(target,q2,K2)
            match=[(v,w) for v,w in self.adj[env] if v==nxt]
            if not match: raise RuntimeError("World branch missing from game")
            cost+=match[0][1]; node=nxt
        raise RuntimeError("Policy did not terminate")

    def policy_worst_regret(self,policy):
        costs={}; regrets={}
        for w in self.worlds:
            c,_=self.simulate_policy(policy,w); o=self.oracle_cost(w)
            costs[w]=c; regrets[w]=c-o
        return max(regrets.values()),costs,regrets
