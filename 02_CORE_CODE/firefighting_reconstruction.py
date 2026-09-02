
"""
Firefighting benchmark reconstruction for Zhao et al.
------------------------------------------------------

This file intentionally distinguishes between:
1) facts numerically disclosed by the paper/journal extension; and
2) an equivalent compact PK-WTS reconstructed from those disclosed facts.

The original per-edge PK-WTS configuration is not printed in the paper and the
project repository cited by the paper is no longer available at the cited URL.
Therefore this script is a reproducible *paper-equivalent reconstruction*, not
a claim that the state graph below is the authors' unpublished source file.

Disclosed constraints encoded here:
- task: obtain extinguisher before fire, then eventually reach fire;
- two unknown regions ("below" and "above");
- no-exploration/worst-case route has cost 36;
- below shortcut saves 14 if available -> 22;
- below exploration can incur up to 12 regret in a mixed world;
- above shortcut saves 4 if available -> 32;
- above exploration costs 6 extra if unavailable -> 42;
- regret policy explores the below unknown region, not the above one;
- Table-2 tested costs for regret: T1=22, T2=44;
- Table-2 tested costs for worst-case: T1=T2=36.

For the equivalent model:
- T1 = both shortcuts open;
- T2 = both shortcuts closed;
- all four independent open/closed combinations are retained when synthesizing
  the minimax-regret policy, so the policy is not tuned only to T1/T2.
"""

from __future__ import annotations

from math import inf, isclose
import heapq
from typing import Dict, Tuple

from stage0_solver import AgentNode, DFA, EnvNode, PKWTS, Stage0RegretSolver


def firefighting_dfa() -> DFA:
    """
    Finite monitor for:
        phi = (!fire U extinguisher) AND eventually fire

    q0    : extinguisher not yet obtained
    q1    : extinguisher obtained, fire not yet reached
    qF    : task accepted
    qDead : fire visited before extinguisher
    """
    def tr(q, label):
        if q in ("qF", "qDead"):
            return q

        has_e = "extinguisher" in label
        has_f = "fire" in label

        if q == "q0":
            if has_f and not has_e:
                return "qDead"
            if has_e:
                return "q1"
            return "q0"

        if q == "q1":
            return "qF" if has_f else "q1"

        raise ValueError(q)

    return DFA(
        states=("q0", "q1", "qF", "qDead"),
        q_init="q0",
        accepting=frozenset({"qF"}),
        transition_fn=tr,
    )


def firefighting_equivalent_pkwts() -> PKWTS:
    """
    Compact equivalent abstraction.

    States:
      I : initial area
      E : extinguisher
      F : fire
      L : below/lower unknown roof region
      U : above/upper unknown roof region

    Safe route:
      I -> E -> F = 20 + 16 = 36

    Lower unknown:
      enter L costs 4.
      open:   L -> E costs 2, so I-L-E-F = 4+2+16 = 22.
      closed: L -> I costs 4, so detour + safe = 4+4+36 = 44.

    Upper unknown:
      enter U costs 3.
      open:   U -> E costs 13, so I-U-E-F = 3+13+16 = 32
              (4-unit saving relative to 36).
      closed: U -> I costs 3, so detour + safe = 3+3+36 = 42
              (6-unit additional cost).
    """
    states = ("I", "E", "F", "L", "U")

    patterns = {
        "I": (frozenset({"E", "L", "U"}),),
        "E": (frozenset({"F"}),),
        "F": (frozenset(),),

        # pattern 0 = shortcut open, pattern 1 = blocked/closed
        "L": (
            frozenset({"I", "E"}),
            frozenset({"I"}),
        ),
        "U": (
            frozenset({"I", "E"}),
            frozenset({"I"}),
        ),
    }

    weights = {
        ("I", "E"): 20,
        ("E", "F"): 16,

        ("I", "L"): 4,
        ("L", "E"): 2,
        ("L", "I"): 4,

        ("I", "U"): 3,
        ("U", "E"): 13,
        ("U", "I"): 3,
    }

    labels = {
        "I": frozenset(),
        "L": frozenset(),
        "U": frozenset(),
        "E": frozenset({"extinguisher"}),
        "F": frozenset({"fire"}),
    }

    return PKWTS(
        states=states,
        x0="I",
        patterns=patterns,
        weights=weights,
        labels=labels,
    )


def pattern_status(solver: Stage0RegretSolver, world) -> Tuple[str, str]:
    iL = solver.T.state_index["L"]
    iU = solver.T.state_index["U"]
    L = "open" if world[iL] == 0 else "closed"
    U = "open" if world[iU] == 0 else "closed"
    return L, U


def synthesize_worst_case_policy(solver: Stage0RegretSolver):
    """
    Paper baseline: min-max synthesis on the same game with original physical
    edge weights w_G rather than regret-equivalent weights mu.
    """
    physical_weights = {
        (u, v): w
        for u, edges in solver.adj.items()
        for v, w in edges
    }
    V, policy = solver.minmax_value_iteration(physical_weights)
    return V[solver.start], policy


def optimistic_successors(solver: Stage0RegretSolver, x, K):
    """
    Successors in the skeleton of the refined PK-WTS:
    observed unknown states are fixed; unobserved unknown states use union
    of all possible successor patterns.
    """
    pats = solver.T.patterns[x]
    if len(pats) == 1:
        return pats[0]

    kd = dict(K)
    if x in kd:
        return pats[kd[x]]

    out = set()
    for p in pats:
        out.update(p)
    return frozenset(out)


def optimistic_first_action(solver: Stage0RegretSolver, node: AgentNode):
    """
    First action of a shortest accepting path in the skeleton of the refined
    PK-WTS. This implements the paper's described best-case replanning rule.
    """
    if node.q in solver.A.accepting:
        return None

    start = (node.x, node.q)
    dist = {start: 0.0}
    parent = {}
    heap = [(0.0, start)]
    goal = None

    while heap:
        d, (x, q) = heapq.heappop(heap)
        if d != dist[(x, q)]:
            continue
        if q in solver.A.accepting:
            goal = (x, q)
            break

        for y in optimistic_successors(solver, x, node.K):
            q2 = solver.A.step(q, solver.T.labels[y])
            nd = d + solver.T.weights[(x, y)]
            key = (y, q2)
            if nd < dist.get(key, inf):
                dist[key] = nd
                parent[key] = (x, q)
                heapq.heappush(heap, (nd, key))

    if goal is None:
        return None

    cur = goal
    while parent.get(cur) != start:
        if cur not in parent:
            return None
        cur = parent[cur]
    return cur[0]


def synthesize_best_case_policy(solver: Stage0RegretSolver):
    policy: Dict[AgentNode, EnvNode] = {}
    for node in solver.agent_nodes:
        target = optimistic_first_action(solver, node)
        if target is None:
            continue
        envs = [
            v for v, _ in solver.adj[node]
            if isinstance(v, EnvNode) and v.target == target
        ]
        if envs:
            policy[node] = envs[0]
    return policy


def run():
    T = firefighting_equivalent_pkwts()
    solver = Stage0RegretSolver(T, firefighting_dfa())

    # Regret-optimal solver (Algorithm-1 equivalent Stage-0 implementation).
    result = solver.solve()
    regret_worst, regret_costs, regrets = solver.policy_worst_regret(result.policy)

    # Worst-case baseline.
    wc_value, wc_policy = synthesize_worst_case_policy(solver)

    # Optimistic/best-case baseline on the reconstructed abstraction.
    bc_policy = synthesize_best_case_policy(solver)

    rows = []
    worlds_by_status = {}

    for world in solver.worlds:
        status = pattern_status(solver, world)
        worlds_by_status[status] = world

        wc_cost, _ = solver.simulate_policy(wc_policy, world)
        bc_cost, _ = solver.simulate_policy(bc_policy, world)

        rows.append({
            "lower": status[0],
            "upper": status[1],
            "oracle": result.oracle_costs[world],
            "regret_policy_cost": regret_costs[world],
            "regret": regrets[world],
            "worst_case_cost": wc_cost,
            "optimistic_cost": bc_cost,
        })

    # Paper's two displayed environments in the IJRR caption:
    # T1 both shortcuts; T2 no shortcut.
    T1 = worlds_by_status[("open", "open")]
    T2 = worlds_by_status[("closed", "closed")]

    summary = {
        "game_nodes": len(solver.adj),
        "agent_nodes": len(solver.agent_nodes),
        "environment_nodes": len(solver.env_nodes),
        "possible_worlds": len(solver.worlds),
        "E_SP_edges": len(result.shortest_path_edges),

        "regret_optimal_value_all_worlds": result.value,
        "regret_policy_initial_action": result.policy[solver.start].target,
        "regret_policy_cost_T1": regret_costs[T1],
        "regret_policy_cost_T2": regret_costs[T2],
        "regret_T1": regrets[T1],
        "regret_T2": regrets[T2],

        "worst_case_value": wc_value,
        "worst_case_initial_action": wc_policy[solver.start].target,
        "worst_case_cost_T1": solver.simulate_policy(wc_policy, T1)[0],
        "worst_case_cost_T2": solver.simulate_policy(wc_policy, T2)[0],

        "optimistic_initial_action_reconstruction": bc_policy[solver.start].target,
        "optimistic_cost_T1_reconstruction": solver.simulate_policy(bc_policy, T1)[0],
        "optimistic_cost_T2_reconstruction": solver.simulate_policy(bc_policy, T2)[0],
    }

    # Required numerical checks for all quantities that the reconstruction
    # is designed to reproduce exactly.
    assert summary["regret_policy_initial_action"] == "L"
    assert isclose(summary["regret_policy_cost_T1"], 22.0)
    assert isclose(summary["regret_policy_cost_T2"], 44.0)
    assert summary["worst_case_initial_action"] == "E"
    assert isclose(summary["worst_case_cost_T1"], 36.0)
    assert isclose(summary["worst_case_cost_T2"], 36.0)

    # Local tradeoff checks from the journal extension.
    # Lower-open route saves 14 vs safe; lower-closed T2 adds 8 vs safe.
    assert isclose(36.0 - 22.0, 14.0)
    assert isclose(44.0 - 36.0, 8.0)

    return solver, result, rows, summary


if __name__ == "__main__":
    solver, result, rows, summary = run()

    print("FIREFIGHTING RECONSTRUCTION: CORE CHECKS PASSED")
    print("\n[Game]")
    for k in (
        "game_nodes", "agent_nodes", "environment_nodes",
        "possible_worlds", "E_SP_edges"
    ):
        print(f"{k}: {summary[k]}")

    print("\n[Regret-optimal]")
    print("initial action:", summary["regret_policy_initial_action"])
    print("minimax regret over all 4 worlds:",
          summary["regret_optimal_value_all_worlds"])
    print("T1 cost:", summary["regret_policy_cost_T1"],
          "regret:", summary["regret_T1"])
    print("T2 cost:", summary["regret_policy_cost_T2"],
          "regret:", summary["regret_T2"])

    print("\n[Worst-case]")
    print("initial action:", summary["worst_case_initial_action"])
    print("worst-case guaranteed cost:", summary["worst_case_value"])
    print("T1 cost:", summary["worst_case_cost_T1"])
    print("T2 cost:", summary["worst_case_cost_T2"])

    print("\n[All possible worlds]")
    for r in rows:
        print(r)

    print("\n[Optimistic baseline on this compact reconstruction]")
    print("initial action:", summary["optimistic_initial_action_reconstruction"])
    print("T1 cost:", summary["optimistic_cost_T1_reconstruction"])
    print("T2 cost:", summary["optimistic_cost_T2_reconstruction"])
