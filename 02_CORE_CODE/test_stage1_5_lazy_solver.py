
from __future__ import annotations

from math import isclose

from dynamic_cvar_regret_solver import DynamicCVaRRegretSolver
from firefighting_reconstruction import firefighting_dfa, firefighting_equivalent_pkwts
from lazy_dynamic_cvar_solver import (
    LazyHorizonDynamicCVaRSolver,
    LazyProgressDynamicCVaRSolver,
    explicit_robust_progress_ranks,
)
from stage0_solver import PKWTS, Stage0RegretSolver
from test_stage1_static_cvar import four_policy_three_world_problem, eventually_goal_dfa


def uniform_prior(T):
    worlds = T.all_worlds()
    return {w: 1.0 / len(worlds) for w in worlds}


def test_three_world_exact_match():
    T, prior, _ = four_policy_three_world_problem()
    dfa = eventually_goal_dfa()

    out = {}
    for alpha, expected_policy, expected_value in [
        (0.0, "C", 3.45),
        (0.75, "A", 7.8),
        (0.90, "D", 9.0),
    ]:
        ref_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha)
        ref = ref_solver.solve()

        lazy_solver = LazyHorizonDynamicCVaRSolver(
            T, dfa, prior, alpha, horizon=2
        )
        lazy = lazy_solver.solve()

        assert ref.policy[ref_solver.game.start].target == expected_policy
        assert lazy.policy[(lazy_solver.start, 2)].target == expected_policy
        assert isclose(ref.dynamic_regret_value, expected_value, abs_tol=1e-8)
        assert isclose(lazy.dynamic_regret_value, expected_value, abs_tol=1e-8)

        out[alpha] = {
            "policy": expected_policy,
            "reference_value": ref.dynamic_regret_value,
            "lazy_value": lazy.dynamic_regret_value,
            "reference_game_nodes": ref.game_nodes,
            "lazy_generated_agent_states": lazy.generated_agent_states,
            "lazy_generated_env_actions": lazy.generated_env_actions,
            "lazy_value_state_budgets": lazy.value_expanded_state_budgets,
            "horizon": 2,
        }
    return out


def test_firefighting_horizon_saturation():
    T = firefighting_equivalent_pkwts()
    dfa = firefighting_dfa()
    prior = uniform_prior(T)

    out = {}
    for alpha in (0.0, 0.5, 0.9):
        ref_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha)
        ref = ref_solver.solve()

        # H=2 admits only the shortest robust safe completion, so it is conservative.
        h2_solver = LazyHorizonDynamicCVaRSolver(T, dfa, prior, alpha, horizon=2)
        h2 = h2_solver.solve()
        assert h2.policy[(h2_solver.start, 2)].target == "E"
        assert h2.dynamic_regret_value >= ref.dynamic_regret_value - 1e-8

        # H=4 admits the lower-region exploratory detour and recovers the
        # unrestricted deterministic reference optimum exactly.
        h4_solver = LazyHorizonDynamicCVaRSolver(T, dfa, prior, alpha, horizon=4)
        h4 = h4_solver.solve()

        assert h4.policy[(h4_solver.start, 4)].target == ref.policy[ref_solver.game.start].target
        assert isclose(
            h4.dynamic_regret_value,
            ref.dynamic_regret_value,
            rel_tol=1e-8,
            abs_tol=1e-8,
        )

        out[alpha] = {
            "reference_policy": ref.policy[ref_solver.game.start].target,
            "reference_value": ref.dynamic_regret_value,
            "H2_policy": h2.policy[(h2_solver.start, 2)].target,
            "H2_value": h2.dynamic_regret_value,
            "H4_policy": h4.policy[(h4_solver.start, 4)].target,
            "H4_value": h4.dynamic_regret_value,
            "reference_game_nodes": ref.game_nodes,
            "H4_lazy_agent_states": h4.generated_agent_states,
            "H4_lazy_env_actions": h4.generated_env_actions,
            "H4_value_state_budgets": h4.value_expanded_state_budgets,
        }
    return out


def test_rank_certificate_on_explicit_game():
    T = firefighting_equivalent_pkwts()
    dfa = firefighting_dfa()
    prior = uniform_prior(T)

    ref_solver = DynamicCVaRRegretSolver(T, dfa, prior, alpha=0.75)
    rank, prog = explicit_robust_progress_ranks(ref_solver.game)

    assert ref_solver.game.start in rank
    r0 = rank[ref_solver.game.start]
    assert r0 == 2

    checked = 0
    for a, actions in prog.items():
        for env in actions:
            for ch, _ in ref_solver.game.adj[env]:
                assert ch in rank
                assert rank[ch] < rank[a]
                checked += 1

    assert checked > 0
    return {
        "minimum_robust_rank": r0,
        "ranked_agent_states": len(rank),
        "checked_progress_branches": checked,
    }


def test_minimal_rank_vs_horizon_policy_class():
    """
    Demonstrates why Stage 1.5 uses Pi_H rather than only minimal-rank actions.

    Minimal-rank progress at firefighting start has rank 2 and forces safe E.
    The H=4 class permits the exploratory L detour and recovers the reference
    dynamic-risk optimum.
    """
    T = firefighting_equivalent_pkwts()
    prior = uniform_prior(T)

    p = LazyProgressDynamicCVaRSolver(
        T, firefighting_dfa(), prior, alpha=0.75, max_rank=8
    )
    pr = p.solve()
    assert pr.start_rank == 2
    assert pr.policy[p.start].target == "E"

    h = LazyHorizonDynamicCVaRSolver(
        T, firefighting_dfa(), prior, alpha=0.75, horizon=4
    )
    hr = h.solve()
    assert hr.policy[(h.start, 4)].target == "L"
    assert hr.dynamic_regret_value < pr.dynamic_regret_value

    return {
        "minimal_rank_policy": pr.policy[p.start].target,
        "minimal_rank_value": pr.dynamic_regret_value,
        "H4_policy": hr.policy[(h.start, 4)].target,
        "H4_value": hr.dynamic_regret_value,
    }


def test_reachable_tied_actions_use_canonical_order():
    states = ("s", "m", "a", "b", "g")
    patterns = {
        "s": (frozenset({"m"}),),
        "m": (frozenset({"a", "b"}),),
        "a": (frozenset({"g"}),),
        "b": (frozenset({"g"}),),
        "g": (frozenset(),),
    }
    weights = {
        ("s", "m"): 1.0,
        ("m", "a"): 1.0,
        ("m", "b"): 1.0,
        ("a", "g"): 1.0,
        ("b", "g"): 1.0,
    }
    labels = {state: frozenset() for state in states}
    labels["g"] = frozenset({"g"})
    world = (0, 0, 0, 0, 0)
    T = PKWTS(states, "s", patterns, weights, labels, (world,))

    def make_solver():
        solver = LazyHorizonDynamicCVaRSolver(
            T, eventually_goal_dfa(), {world: 1.0}, alpha=0.5, horizon=3
        )
        original_prebranch = solver.prebranch_action_lower_bound
        original_action = solver.action_lower_bound
        solver.prebranch_action_lower_bound = lambda env: (
            0.0 if env.x == "m" and env.target == "b"
            else 2.0 if env.x == "m" and env.target == "a"
            else original_prebranch(env)
        )
        solver.action_lower_bound = lambda env: (
            0.0 if env.x == "m" and env.target == "b"
            else 2.0 if env.x == "m" and env.target == "a"
            else original_action(env)
        )
        return solver

    solver = make_solver()
    result = solver.solve()
    start_action = solver.actions(solver.start)[0]
    reached_state = next(iter(solver.branches(start_action)))
    initial_action = result.policy[(reached_state, 2)]

    rerun_solver = make_solver()
    rerun_start_action = rerun_solver.actions(rerun_solver.start)[0]
    rerun_state = next(iter(rerun_solver.branches(rerun_start_action)))
    conditional_value = rerun_solver.budget_value(rerun_state, 2)
    rerun_action = rerun_solver.budget_policy[(rerun_state, 2)]

    q_by_target = {}
    for env in rerun_solver.actions(rerun_state):
        child = next(iter(rerun_solver.branches(env)))
        q_by_target[env.target] = (
            rerun_solver.action_cost(env)
            + rerun_solver.budget_value(child, 1)
        )
    optimal_targets = {
        target for target, value in q_by_target.items()
        if isclose(value, conditional_value, rel_tol=0.0, abs_tol=1e-12)
    }

    assert optimal_targets == {"a", "b"}
    assert initial_action.target in optimal_targets
    assert rerun_action.target in optimal_targets
    assert initial_action.target == rerun_action.target == "a"
    assert isclose(conditional_value, 2.0, rel_tol=0.0, abs_tol=1e-12)

    return {
        "conditional_value": conditional_value,
        "optimal_targets": tuple(sorted(optimal_targets)),
        "canonical_action": rerun_action.target,
    }


if __name__ == "__main__":
    a = test_three_world_exact_match()
    b = test_firefighting_horizon_saturation()
    c = test_rank_certificate_on_explicit_game()
    d = test_minimal_rank_vs_horizon_policy_class()
    e = test_reachable_tied_actions_use_canonical_order()

    print("STAGE 1.5 CORE TESTS: ALL PASSED")
    print("\n[Three-world exact match]")
    for k, v in a.items():
        print(k, v)
    print("\n[Firefighting horizon saturation]")
    for k, v in b.items():
        print(k, v)
    print("\n[Rank certificate]")
    for k, v in c.items():
        print(k, v)
    print("\n[Why Pi_H is needed]")
    for k, v in d.items():
        print(k, v)
    print("\n[Reachable tied-action re-solve]")
    for k, v in e.items():
        print(k, v)
