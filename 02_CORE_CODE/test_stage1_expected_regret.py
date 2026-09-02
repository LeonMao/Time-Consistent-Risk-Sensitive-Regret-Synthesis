from __future__ import annotations

from math import isclose

from stage0_solver import DFA, PKWTS, Stage0RegretSolver
from stage1_common import BeliefModel, normalize_prior, restrict_to_prior_support
from expected_regret_solver import BayesianExpectedRegretSolver
from firefighting_reconstruction import firefighting_dfa, firefighting_equivalent_pkwts


def eventually_goal_dfa():
    def tr(q, label):
        if q == "qF":
            return "qF"
        return "qF" if "goal" in label else "qN"
    return DFA(("qN", "qF"), "qN", frozenset({"qF"}), tr)


def shortcut_problem():
    states = ("0", "1", "2", "3", "4", "5")
    patterns = {
        "0": (frozenset({"1", "2"}),),
        "1": (frozenset({"3"}),),
        "2": (frozenset({"0", "5"}), frozenset({"0"})),  # open / closed
        "3": (frozenset({"4"}),),
        "4": (frozenset({"5"}),),
        "5": (frozenset(),),
    }
    weights = {
        ("0", "1"): 2,
        ("0", "2"): 1,
        ("1", "3"): 2,
        ("2", "0"): 1,
        ("2", "5"): 2,
        ("3", "4"): 3,
        ("4", "5"): 3,
    }
    labels = {x: frozenset() for x in states}
    labels["5"] = frozenset({"goal"})
    return PKWTS(states, "0", patterns, weights, labels)


def unsafe_temptation_problem():
    # High-probability risky shortcut reaches goal cheaply if open, but if closed
    # it reaches a nonaccepting dead-end. Robust scLTL satisfaction must reject it.
    states = ("s", "u", "a", "g")
    patterns = {
        "s": (frozenset({"u", "a"}),),
        "u": (frozenset({"g"}), frozenset()),  # open / closed dead-end
        "a": (frozenset({"g"}),),
        "g": (frozenset(),),
    }
    weights = {
        ("s", "u"): 1,
        ("u", "g"): 1,
        ("s", "a"): 5,
        ("a", "g"): 5,
    }
    labels = {x: frozenset() for x in states}
    labels["g"] = frozenset({"goal"})
    return PKWTS(states, "s", patterns, weights, labels)


def world_by_pattern(T, state, pattern_index):
    idx = T.state_index[state]
    return next(w for w in T.all_worlds() if w[idx] == pattern_index)


def test_belief_update_shortcut():
    T = shortcut_problem()
    ow = world_by_pattern(T, "2", 0)
    cw = world_by_pattern(T, "2", 1)
    prior = normalize_prior(T, {ow: 0.6, cw: 0.4})
    restricted = restrict_to_prior_support(T, prior)
    game = Stage0RegretSolver(restricted, eventually_goal_dfa())
    game.build_game()
    belief = BeliefModel(prior)

    # At the initial choice 0->2, the observation branches inherit 0.6/0.4.
    env = next(v for v, _ in game.adj[game.start] if v.target == "2")
    dist = belief.observation_distribution(game, env)
    by_pattern = {dict(child.K)["2"]: p for child, p in dist.items()}
    assert isclose(by_pattern[0], 0.6)
    assert isclose(by_pattern[1], 0.4)

    # After observing open, posterior collapses to the open world.
    open_child = next(child for child in dist if dict(child.K)["2"] == 0)
    post = belief.posterior(game, open_child.K)
    assert post == {ow: 1.0}


def test_shortcut_probability_switch():
    T = shortcut_problem()
    ow = world_by_pattern(T, "2", 0)
    cw = world_by_pattern(T, "2", 1)

    # p(open)=0.6 -> explore.
    s_hi = BayesianExpectedRegretSolver(T, eventually_goal_dfa(), {ow: 0.6, cw: 0.4})
    r_hi = s_hi.solve()
    assert r_hi.policy[s_hi.game.start].target == "2"
    assert isclose(r_hi.expected_cost, 6.6)
    assert isclose(r_hi.expected_oracle_cost, 5.8)
    assert isclose(r_hi.expected_regret, 0.8)
    assert isclose(r_hi.expected_regret, r_hi.expected_cost - r_hi.expected_oracle_cost)

    # p(open)=0.1 -> safe route.
    s_lo = BayesianExpectedRegretSolver(T, eventually_goal_dfa(), {ow: 0.1, cw: 0.9})
    r_lo = s_lo.solve()
    assert r_lo.policy[s_lo.game.start].target == "1"
    assert isclose(r_lo.expected_cost, 10.0)
    assert isclose(r_lo.expected_oracle_cost, 9.3)
    assert isclose(r_lo.expected_regret, 0.7)

    # Analytic threshold: explore iff 3p+12(1-p) < 10 -> p > 2/9.
    threshold = 2.0 / 9.0
    assert 0.1 < threshold < 0.6

    return {
        "threshold_open_probability": threshold,
        "high_prior_action": r_hi.policy[s_hi.game.start].target,
        "high_prior_expected_cost": r_hi.expected_cost,
        "high_prior_expected_regret": r_hi.expected_regret,
        "low_prior_action": r_lo.policy[s_lo.game.start].target,
        "low_prior_expected_cost": r_lo.expected_cost,
        "low_prior_expected_regret": r_lo.expected_regret,
    }


def test_robust_scLTL_constraint():
    T = unsafe_temptation_problem()
    ow = world_by_pattern(T, "u", 0)
    cw = world_by_pattern(T, "u", 1)

    # Even with 99% chance that the cheap shortcut is open, the risky action is
    # not robustly task-satisfying because the 1% closed world is a dead-end.
    solver = BayesianExpectedRegretSolver(T, eventually_goal_dfa(), {ow: 0.99, cw: 0.01})
    result = solver.solve()
    assert result.policy[solver.game.start].target == "a"
    assert isclose(result.expected_cost, 10.0)
    return {
        "risky_open_probability": 0.99,
        "chosen_action": result.policy[solver.game.start].target,
        "expected_cost": result.expected_cost,
    }


def test_firefighting_uniform_prior():
    T = firefighting_equivalent_pkwts()
    worlds = T.all_worlds()
    prior = {w: 0.25 for w in worlds}

    solver = BayesianExpectedRegretSolver(T, firefighting_dfa(), prior)
    result = solver.solve()

    # Under the compact paper-equivalent reconstruction and uniform prior,
    # exploring the lower region minimizes expected cost among robust policies.
    assert result.policy[solver.game.start].target == "L"
    assert isclose(result.expected_cost, 33.0)
    assert isclose(result.expected_oracle_cost, 28.0)
    assert isclose(result.expected_regret, 5.0)
    assert abs(result.evaluation["equivalence_residual"]) < 1e-10

    return {
        "game_nodes": result.game_nodes,
        "winning_nodes": result.winning_nodes,
        "initial_action": result.policy[solver.game.start].target,
        "expected_cost": result.expected_cost,
        "expected_oracle_cost": result.expected_oracle_cost,
        "expected_regret": result.expected_regret,
    }


if __name__ == "__main__":
    test_belief_update_shortcut()
    shortcut = test_shortcut_probability_switch()
    safe = test_robust_scLTL_constraint()
    fire = test_firefighting_uniform_prior()

    print("STAGE 1.2: ALL TESTS PASSED")
    print("\n[Shortcut probability switch]")
    for k, v in shortcut.items():
        print(f"{k}: {v}")
    print("\n[Robust scLTL constraint]")
    for k, v in safe.items():
        print(f"{k}: {v}")
    print("\n[Firefighting / uniform prior]")
    for k, v in fire.items():
        print(f"{k}: {v}")
