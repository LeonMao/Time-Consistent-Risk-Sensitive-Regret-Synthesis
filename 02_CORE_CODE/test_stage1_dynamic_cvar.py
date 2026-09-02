
from __future__ import annotations

from math import isclose

from dynamic_cvar_regret_solver import DynamicCVaRRegretSolver
from expected_regret_solver import BayesianExpectedRegretSolver
from firefighting_reconstruction import firefighting_dfa, firefighting_equivalent_pkwts
from stage0_solver import Stage0RegretSolver
from stage1_3_risk import discrete_cvar
from test_stage1_static_cvar import (
    eventually_goal_dfa,
    four_policy_three_world_problem,
)


def uniform_prior(T):
    worlds = T.all_worlds()
    return {w: 1.0 / len(worlds) for w in worlds}


def test_three_world_endpoints_and_midpoint():
    T, prior, worlds = four_policy_three_world_problem()
    dfa = eventually_goal_dfa()

    # alpha = 0: nested CVaR collapses to expectation -> Stage 1.2 policy C.
    dyn0_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.0)
    dyn0 = dyn0_solver.solve()
    assert dyn0.policy[dyn0_solver.game.start].target == "C"
    assert isclose(dyn0.dynamic_regret_value, 3.45, abs_tol=1e-8)

    # alpha = .75: one observation fully reveals the world, so dynamic and static
    # CVaR coincide on this one-revelation benchmark -> policy A, CVaR=7.8.
    dyn75_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.75)
    dyn75 = dyn75_solver.solve()
    assert dyn75.policy[dyn75_solver.game.start].target == "A"
    assert isclose(dyn75.dynamic_regret_value, 7.8, abs_tol=1e-8)

    # Dynamic finite-support threshold:
    # local initial branch probabilities are (.60,.25,.15), so p_min=.15.
    pmin = dyn75_solver.dynamic_min_positive_probability()
    assert isclose(pmin, 0.15, abs_tol=1e-12)

    # alpha=.90 is beyond 1-p_min=.85 -> minimax regret, policy D, value 9.
    dyn90_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.90)
    dyn90 = dyn90_solver.solve()
    assert dyn90.policy[dyn90_solver.game.start].target == "D"
    assert isclose(dyn90.dynamic_regret_value, 9.0, abs_tol=1e-8)

    # Compare to frozen Stage 0 minimax solver.
    mr_solver = Stage0RegretSolver(T, dfa)
    mr = mr_solver.solve()
    assert mr.policy[mr_solver.start].target == "D"
    assert isclose(mr.value, dyn90.dynamic_regret_value, abs_tol=1e-8)

    return {
        "alpha0_policy": dyn0.policy[dyn0_solver.game.start].target,
        "alpha0_value": dyn0.dynamic_regret_value,
        "alpha075_policy": dyn75.policy[dyn75_solver.game.start].target,
        "alpha075_value": dyn75.dynamic_regret_value,
        "pmin_dynamic": pmin,
        "alpha090_policy": dyn90.policy[dyn90_solver.game.start].target,
        "alpha090_value": dyn90.dynamic_regret_value,
    }


def test_static_vs_dynamic_time_consistency_counterexample():
    """
    Same Stage-1.3 counterexample, now compute the nested objective explicitly.

    H has probability .05, O has probability .95.
    Conditional regrets:
        H: A=(20,3), B=(16,17), posterior=(.5,.5)
        O: both=(6,9), posterior=(.5,.5)
    alpha=.75.

    Static precommitment:
        A: 10.1 < B: 10.5  -> A

    Dynamic nested:
        rho_H(A)=20, rho_H(B)=17
        rho_O=9
        rho_0(A)=CVaR((20,9),(.05,.95))=11.2
        rho_0(B)=CVaR((17,9),(.05,.95))=10.6
        -> B

    Dynamic choice B is also conditionally optimal at H.
    """
    alpha = 0.75

    probs_global = (0.025, 0.025, 0.475, 0.475)
    A = (20.0, 3.0, 6.0, 9.0)
    B = (16.0, 17.0, 6.0, 9.0)

    static_A = discrete_cvar(A, probs_global, alpha)
    static_B = discrete_cvar(B, probs_global, alpha)
    assert isclose(static_A, 10.1)
    assert isclose(static_B, 10.5)
    assert static_A < static_B

    rho_H_A = discrete_cvar(A[:2], (0.5, 0.5), alpha)
    rho_H_B = discrete_cvar(B[:2], (0.5, 0.5), alpha)
    rho_O = discrete_cvar(A[2:], (0.5, 0.5), alpha)

    nested_A = discrete_cvar((rho_H_A, rho_O), (0.05, 0.95), alpha)
    nested_B = discrete_cvar((rho_H_B, rho_O), (0.05, 0.95), alpha)

    assert isclose(rho_H_A, 20.0)
    assert isclose(rho_H_B, 17.0)
    assert isclose(rho_O, 9.0)
    assert isclose(nested_A, 11.2)
    assert isclose(nested_B, 10.6)
    assert nested_B < nested_A
    assert rho_H_B < rho_H_A

    return {
        "static_precommitment": "A",
        "static_A": static_A,
        "static_B": static_B,
        "dynamic_nested": "B",
        "nested_A": nested_A,
        "nested_B": nested_B,
        "conditional_H": "B",
        "conditional_H_A": rho_H_A,
        "conditional_H_B": rho_H_B,
    }


def test_firefighting_endpoints():
    T = firefighting_equivalent_pkwts()
    dfa = firefighting_dfa()
    prior = uniform_prior(T)

    # alpha=0 must match frozen Stage 1.2 result.
    er = BayesianExpectedRegretSolver(T, dfa, prior).solve()
    dyn0_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.0)
    dyn0 = dyn0_solver.solve()

    assert dyn0.policy[dyn0_solver.game.start].target == "L"
    assert isclose(dyn0.dynamic_regret_value, er.expected_regret, abs_tol=1e-8)
    assert isclose(dyn0.dynamic_regret_value, 5.0, abs_tol=1e-8)

    # Compute local minimum probability. With uniform prior, accepting without
    # revealing all uncertainty can leave four worlds -> pmin=.25.
    pmin = dyn0_solver.dynamic_min_positive_probability()
    assert isclose(pmin, 0.25, abs_tol=1e-12)

    # alpha=.90 >= .75 threshold -> recover Stage-0 minimax regret.
    dyn90_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.90)
    dyn90 = dyn90_solver.solve()

    mr_solver = Stage0RegretSolver(T, dfa)
    mr = mr_solver.solve()

    assert dyn90.policy[dyn90_solver.game.start].target == "L"
    assert mr.policy[mr_solver.start].target == "L"
    assert isclose(dyn90.dynamic_regret_value, mr.value, abs_tol=1e-8)
    assert isclose(dyn90.dynamic_regret_value, 12.0, abs_tol=1e-8)

    # Mid-risk point should remain finite and between expectation and worst case.
    dyn50_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.50)
    dyn50 = dyn50_solver.solve()
    assert dyn0.dynamic_regret_value <= dyn50.dynamic_regret_value + 1e-8
    assert dyn50.dynamic_regret_value <= dyn90.dynamic_regret_value + 1e-8

    return {
        "game_nodes": dyn0.game_nodes,
        "alpha0_policy": dyn0.policy[dyn0_solver.game.start].target,
        "alpha0_dynamic_regret": dyn0.dynamic_regret_value,
        "alpha05_policy": dyn50.policy[dyn50_solver.game.start].target,
        "alpha05_dynamic_regret": dyn50.dynamic_regret_value,
        "dynamic_pmin": pmin,
        "alpha09_policy": dyn90.policy[dyn90_solver.game.start].target,
        "alpha09_dynamic_regret": dyn90.dynamic_regret_value,
        "stage0_minimax_regret": mr.value,
    }


if __name__ == "__main__":
    e = test_three_world_endpoints_and_midpoint()
    t = test_static_vs_dynamic_time_consistency_counterexample()
    f = test_firefighting_endpoints()

    print("STAGE 1.4: ALL TESTS PASSED")
    print("\n[Expected -> Dynamic CVaR -> Minimax endpoints]")
    for k, v in e.items():
        print(f"{k}: {v}")
    print("\n[Static vs dynamic time consistency]")
    for k, v in t.items():
        print(f"{k}: {v}")
    print("\n[Firefighting]")
    for k, v in f.items():
        print(f"{k}: {v}")
