
from __future__ import annotations

from math import isclose

from expected_regret_solver import BayesianExpectedRegretSolver
from stage0_solver import DFA, EnvNode, PKWTS, Stage0RegretSolver
from stage1_3_risk import (
    cvar_risk_envelope,
    discrete_cvar,
    finite_support_minimax_threshold,
)
from static_cvar_selector import StaticCVaRCandidateSelector


def eventually_goal_dfa():
    def tr(q, label):
        if q == "qF":
            return "qF"
        return "qF" if "g" in label else "qN"
    return DFA(("qN", "qF"), "qN", frozenset({"qF"}), tr)


def four_policy_three_world_problem():
    """
    Direct-choice benchmark engineered so that:
      worst-cost      -> B
      minimax-regret  -> D
      expected-regret -> C
      CVaR_0.75       -> A

    World-wise total costs:
              theta1 theta2 theta3
       A        5      9     16
       B       13     7      5
       C        2     13     18
       D       11     15     13

    Prior: (0.60, 0.25, 0.15)
    Oracle: (2, 7, 5)
    Regrets:
       A = (3, 2, 11)
       B = (11,0,0)
       C = (0, 6, 13)
       D = (9, 8, 8)
    """
    states = ("s", "A", "B", "C", "D", "g1", "g2", "g3")

    patterns = {
        "s": (frozenset({"A", "B", "C", "D"}),),
        "A": (frozenset({"g1"}), frozenset({"g2"}), frozenset({"g3"})),
        "B": (frozenset({"g1"}), frozenset({"g2"}), frozenset({"g3"})),
        "C": (frozenset({"g1"}), frozenset({"g2"}), frozenset({"g3"})),
        "D": (frozenset({"g1"}), frozenset({"g2"}), frozenset({"g3"})),
        "g1": (frozenset(),),
        "g2": (frozenset(),),
        "g3": (frozenset(),),
    }

    weights = {
        ("s", "A"): 1, ("s", "B"): 1, ("s", "C"): 1, ("s", "D"): 1,
        ("A", "g1"): 4,  ("A", "g2"): 8,  ("A", "g3"): 15,
        ("B", "g1"): 12, ("B", "g2"): 6,  ("B", "g3"): 4,
        ("C", "g1"): 1,  ("C", "g2"): 12, ("C", "g3"): 17,
        ("D", "g1"): 10, ("D", "g2"): 14, ("D", "g3"): 12,
    }

    labels = {x: frozenset() for x in states}
    labels["g1"] = frozenset({"g"})
    labels["g2"] = frozenset({"g"})
    labels["g3"] = frozenset({"g"})

    # Correlated worlds: A/B/C/D all reveal the same world index.
    theta1 = (0, 0, 0, 0, 0, 0, 0, 0)
    theta2 = (0, 1, 1, 1, 1, 0, 0, 0)
    theta3 = (0, 2, 2, 2, 2, 0, 0, 0)

    T = PKWTS(
        states=states,
        x0="s",
        patterns=patterns,
        weights=weights,
        labels=labels,
        allowed_worlds=(theta1, theta2, theta3),
    )
    prior = {theta1: 0.60, theta2: 0.25, theta3: 0.15}
    return T, prior, (theta1, theta2, theta3)


def route_policy(game: Stage0RegretSolver, route: str):
    """Complete the only reachable continuation after committing to route."""
    policy = {}

    start_env = next(
        v for v, _ in game.adj[game.start]
        if isinstance(v, EnvNode) and v.target == route
    )
    policy[game.start] = start_env

    # After entering the chosen route, knowledge reveals which goal edge exists.
    for a in game.agent_nodes:
        if a.x != route or a.q in game.A.accepting:
            continue
        envs = [v for v, _ in game.adj[a] if isinstance(v, EnvNode)]
        if len(envs) != 1:
            raise RuntimeError(f"Expected forced continuation at {a}, got {envs}")
        policy[a] = envs[0]

    return policy


def test_four_way_separation():
    T, prior, worlds = four_policy_three_world_problem()
    dfa = eventually_goal_dfa()

    game = Stage0RegretSolver(T, dfa)
    game.build_game()

    policies = {r: route_policy(game, r) for r in ("A", "B", "C", "D")}

    # Stage 0: minimax regret must choose D.
    mr_solver = Stage0RegretSolver(T, dfa)
    mr = mr_solver.solve()
    assert mr.policy[mr_solver.start].target == "D"
    assert isclose(mr.value, 9.0)

    # Stage 1.2: expected regret / expected cost must choose C.
    er_solver = BayesianExpectedRegretSolver(T, dfa, prior)
    er = er_solver.solve()
    assert er.policy[er_solver.game.start].target == "C"
    assert isclose(er.expected_regret, 3.45)

    # Stage 1.3 static CVaR at alpha=.75 must choose A.
    selector = StaticCVaRCandidateSelector(game, prior, alpha=0.75)
    result = selector.solve(policies)
    assert result.selected_name == "A"

    expected = {
        "A": {"regrets": (3, 2, 11), "mean": 3.95, "cvar": 7.8, "max": 11},
        "B": {"regrets": (11, 0, 0), "mean": 6.60, "cvar": 11.0, "max": 11},
        "C": {"regrets": (0, 6, 13), "mean": 3.45, "cvar": 10.2, "max": 13},
        "D": {"regrets": (9, 8, 8), "mean": 8.60, "cvar": 9.0, "max": 9},
    }

    for name, ex in expected.items():
        ev = result.evaluations[name]
        got = tuple(ev.evaluation["regrets"][w] for w in worlds)
        assert got == ex["regrets"]
        assert isclose(ev.expected_regret, ex["mean"])
        assert isclose(ev.cvar_regret, ex["cvar"])
        assert isclose(ev.worst_regret, ex["max"])

    # Risk-envelope interpretation for policy A:
    # p=(.6,.25,.15), alpha=.75 -> q upper bounds=(2.4,1,.6).
    # Worst tail puts q=.6 on regret 11 and remaining .4 on regret 3.
    evA = result.evaluations["A"]
    qA = tuple(round(evA.distorted_world_weights[w], 10) for w in worlds)
    assert qA == (0.4, 0.0, 0.6)

    # Worst-cost (absolute cost) is B, completing the four-way separation.
    worst_cost = {}
    for name, pol in policies.items():
        costs = []
        for w in worlds:
            c, _ = game.simulate_policy(pol, w)
            costs.append(c)
        worst_cost[name] = max(costs)
    assert min(worst_cost, key=worst_cost.get) == "B"

    return {
        "worst_cost_policy": "B",
        "minimax_regret_policy": mr.policy[mr_solver.start].target,
        "expected_regret_policy": er.policy[er_solver.game.start].target,
        "cvar_075_policy": result.selected_name,
        "cvar_A": result.evaluations["A"].cvar_regret,
        "cvar_B": result.evaluations["B"].cvar_regret,
        "cvar_C": result.evaluations["C"].cvar_regret,
        "cvar_D": result.evaluations["D"].cvar_regret,
        "risk_envelope_A": qA,
    }


def test_endpoint_and_finite_threshold():
    T, prior, worlds = four_policy_three_world_problem()
    game = Stage0RegretSolver(T, eventually_goal_dfa())
    game.build_game()
    policies = {r: route_policy(game, r) for r in ("A", "B", "C", "D")}

    # alpha = 0 exactly equals expectation and therefore selects C.
    r0 = StaticCVaRCandidateSelector(game, prior, alpha=0.0).solve(policies)
    assert r0.selected_name == "C"
    for ev in r0.evaluations.values():
        assert isclose(ev.cvar_regret, ev.expected_regret)

    # Finite-support theorem: alpha >= 1 - p_min = .85 implies CVaR=max regret
    # for every policy, not just asymptotically.
    alpha_crit = finite_support_minimax_threshold(prior)
    assert isclose(alpha_crit, 0.85)

    r90 = StaticCVaRCandidateSelector(game, prior, alpha=0.90).solve(policies)
    assert r90.selected_name == "D"
    for ev in r90.evaluations.values():
        assert isclose(ev.cvar_regret, ev.worst_regret)

    # Monotonicity in alpha for each fixed policy.
    grid = (0.0, 0.25, 0.50, 0.75, 0.85, 0.90)
    for name, pol in policies.items():
        vals = [
            StaticCVaRCandidateSelector(game, prior, a).solve({name: pol})
            .selected_evaluation.cvar_regret
            for a in grid
        ]
        assert all(vals[i] <= vals[i+1] + 1e-10 for i in range(len(vals)-1))

    return {
        "alpha_zero_policy": r0.selected_name,
        "alpha_critical": alpha_crit,
        "alpha_090_policy": r90.selected_name,
    }


def test_static_cvar_time_inconsistency():
    """
    Minimal two-stage regret-tree counterexample.

    A rare observation branch H has total probability .05:
        theta1=.025, theta2=.025
    The other branch has:
        theta3=.475, theta4=.475

    At H the robot can continue with action A or B.
    The regrets in H are:
        A: (20,3)
        B: (16,17)
    Outside H, both have the same regrets:
        (6,9)

    With alpha=.75:
      global precommitment:
          CVaR(A)=10.1 < CVaR(B)=10.5  -> choose A
      after H occurs, posterior=(.5,.5):
          CVaR(A|H)=20 > CVaR(B|H)=17 -> re-optimize to B

    Hence the continuation of the ex-ante static-CVaR optimum is not
    conditionally optimal: static CVaR is time-inconsistent.
    """
    alpha = 0.75
    probs = (0.025, 0.025, 0.475, 0.475)
    loss_A = (20.0, 3.0, 6.0, 9.0)
    loss_B = (16.0, 17.0, 6.0, 9.0)

    global_A = discrete_cvar(loss_A, probs, alpha)
    global_B = discrete_cvar(loss_B, probs, alpha)
    assert isclose(global_A, 10.1)
    assert isclose(global_B, 10.5)
    assert global_A < global_B

    post_H = (0.5, 0.5)
    cond_A = discrete_cvar(loss_A[:2], post_H, alpha)
    cond_B = discrete_cvar(loss_B[:2], post_H, alpha)
    assert isclose(cond_A, 20.0)
    assert isclose(cond_B, 17.0)
    assert cond_B < cond_A

    return {
        "alpha": alpha,
        "global_precommit_action": "A",
        "global_cvar_A": global_A,
        "global_cvar_B": global_B,
        "conditional_reoptimized_action_after_H": "B",
        "conditional_cvar_A": cond_A,
        "conditional_cvar_B": cond_B,
    }


def test_primal_dual_cvar():
    values = (3.0, 2.0, 11.0)
    probs = (0.60, 0.25, 0.15)
    alpha = 0.75
    primal = discrete_cvar(values, probs, alpha)
    dual, q = cvar_risk_envelope(values, probs, alpha)
    assert isclose(primal, 7.8)
    assert isclose(dual, 7.8)
    assert tuple(round(x, 10) for x in q) == (0.4, 0.0, 0.6)
    return {"primal": primal, "dual": dual, "q": tuple(q)}


if __name__ == "__main__":
    sep = test_four_way_separation()
    endpoint = test_endpoint_and_finite_threshold()
    tic = test_static_cvar_time_inconsistency()
    dual = test_primal_dual_cvar()

    print("STAGE 1.3: ALL TESTS PASSED")
    print("\n[Four-way policy separation]")
    for k, v in sep.items():
        print(f"{k}: {v}")
    print("\n[Endpoints and finite minimax threshold]")
    for k, v in endpoint.items():
        print(f"{k}: {v}")
    print("\n[Static CVaR time inconsistency]")
    for k, v in tic.items():
        print(f"{k}: {v}")
    print("\n[CVaR primal / risk-envelope dual]")
    for k, v in dual.items():
        print(f"{k}: {v}")
