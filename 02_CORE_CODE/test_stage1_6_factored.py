
from __future__ import annotations

from collections import defaultdict
from math import isclose

from factored_belief import MixtureProductPrior, ProductComponent
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage0_solver import DFA, PKWTS, Stage0RegretSolver
from symbolic_oracle import MonotoneBinaryOracleDiagram


def eventually_goal_dfa():
    def tr(q, label):
        if q == "qF":
            return "qF"
        return "qF" if "goal" in label else "qN"
    return DFA(("qN", "qF"), "qN", frozenset({"qF"}), tr)


def make_multishortcut(m: int, safe_cost: float = 30.0) -> PKWTS:
    """
    Hub benchmark:
        s -> g : safe route, cost safe_cost
        s -> u_i : inspect shortcut i
        u_i closed : only return to s
        u_i open   : may return to s or go to g

    Open total costs increase with i, so u_0 is the best possible shortcut.
    This ordering lets the symbolic oracle collapse all lower-priority variables
    once an earlier shortcut is known open.
    """
    if m < 1:
        raise ValueError("m must be >=1")

    unknowns = tuple(f"u{i}" for i in range(m))
    states = ("s",) + unknowns + ("g",)

    patterns = {
        "s": (frozenset(set(unknowns) | {"g"}),),
        "g": (frozenset(),),
    }
    weights = {("s", "g"): float(safe_cost)}
    labels = {x: frozenset() for x in states}
    labels["g"] = frozenset({"goal"})

    for i, u in enumerate(unknowns):
        # closed ⊂ open
        patterns[u] = (
            frozenset({"s"}),
            frozenset({"s", "g"}),
        )
        entry = 1.0
        back = 1.0
        exit_cost = 2.0 + 0.35 * i
        weights[("s", u)] = entry
        weights[(u, "s")] = back
        weights[(u, "g")] = exit_cost

    return PKWTS(
        states=states,
        x0="s",
        patterns=patterns,
        weights=weights,
        labels=labels,
    )


def independent_prior(T: PKWTS, p_open: float = 0.25) -> MixtureProductPrior:
    marg = {}
    for x in T.states:
        if len(T.patterns[x]) > 1:
            marg[x] = (1.0 - p_open, p_open)  # closed, open
    return MixtureProductPrior(
        T,
        [ProductComponent(1.0, marg)],
    )


def correlated_prior(T: PKWTS) -> MixtureProductPrior:
    """
    Two latent topology regimes:
      good-connectivity mode: shortcuts likely open
      poor-connectivity mode: shortcuts unlikely open

    The latent mode creates positive correlation among shortcut states.
    """
    good = {}
    poor = {}
    for x in T.states:
        if len(T.patterns[x]) > 1:
            good[x] = (0.15, 0.85)
            poor[x] = (0.90, 0.10)

    return MixtureProductPrior(
        T,
        [
            ProductComponent(0.40, good),
            ProductComponent(0.60, poor),
        ],
    )


def explicit_oracle_distribution(T, dfa, prior):
    engine = Stage0RegretSolver(T, dfa)
    agg = defaultdict(float)
    explicit = prior.explicit_prior_for_validation()
    for w, p in explicit.items():
        agg[engine.oracle_cost(w)] += p
    return dict(agg), explicit


def assert_distributions_close(a, b, tol=1e-10):
    keys = set(a) | set(b)
    for k in keys:
        assert isclose(a.get(k, 0.0), b.get(k, 0.0), abs_tol=tol), (k, a, b)


def test_correlated_posterior_without_world_enumeration():
    T = make_multishortcut(2)
    prior = correlated_prior(T)

    K0 = frozenset()
    # Marginal P(open) = .4*.85 + .6*.10 = .40
    p_u0 = prior.marginal(K0, "u0")[1]
    p_u1 = prior.marginal(K0, "u1")[1]
    assert isclose(p_u0, 0.40)
    assert isclose(p_u1, 0.40)

    # Observe u0=open.
    K1 = prior.condition(K0, "u0", 1)
    evidence_masses = prior.component_evidence_masses(K1)
    assert isclose(evidence_masses[0], 0.34)
    assert isclose(evidence_masses[1], 0.06)
    mode_post = prior.component_posterior_weights(K1)
    # Posterior mode-good = .4*.85 / (.4*.85 + .6*.10) = .85
    assert isclose(mode_post[0], 0.85)
    assert isclose(mode_post[1], 0.15)

    # Correlation: P(u1=open | u0=open)=.85*.85+.15*.10=.7375
    cond = prior.marginal(K1, "u1")[1]
    assert isclose(cond, 0.7375)

    # Explicit enumeration is used only to verify the factored formulas.
    ep = prior.explicit_prior_for_validation()
    joint_open = 0.0
    u0_open = 0.0
    idx = T.state_index
    for w, p in ep.items():
        if w[idx["u0"]] == 1:
            u0_open += p
            if w[idx["u1"]] == 1:
                joint_open += p
    assert isclose(joint_open / u0_open, cond)

    return {
        "marginal_open": p_u0,
        "component_evidence_mass": sum(evidence_masses),
        "posterior_good_mode_after_u0_open": mode_post[0],
        "P_u1_open_given_u0_open": cond,
        "explicit_worlds_used_only_for_check": len(ep),
    }


def test_zero_mass_assignments_are_omitted():
    T = make_multishortcut(1)
    prior = MixtureProductPrior(
        T,
        [ProductComponent(1.0, {"u0": (1.0, 0.0)})],
    )

    assert prior.observation_distribution(frozenset(), "u0") == {0: 1.0}
    try:
        prior.condition(frozenset(), "u0", 1)
    except ValueError as exc:
        assert "zero probability" in str(exc)
    else:
        raise AssertionError("Zero-mass evidence must be rejected.")

    symbolic = MonotoneBinaryOracleDiagram(T, eventually_goal_dfa(), prior)
    dist_sym = symbolic.oracle_cost_distribution(frozenset())
    dist_exp, _ = explicit_oracle_distribution(T, eventually_goal_dfa(), prior)
    assert_distributions_close(dist_sym, dist_exp)

    return {
        "positive_observation_branches": 1,
        "oracle_support_values": len(dist_sym),
    }


def test_symbolic_oracle_matches_explicit():
    dfa = eventually_goal_dfa()
    out = {}

    for label, prior_fn in [
        ("independent", lambda T: independent_prior(T, 0.25)),
        ("correlated", correlated_prior),
    ]:
        T = make_multishortcut(7)
        prior = prior_fn(T)
        symbolic = MonotoneBinaryOracleDiagram(T, dfa, prior)

        dist_sym = symbolic.oracle_cost_distribution(frozenset())
        dist_exp, ep = explicit_oracle_distribution(T, dfa, prior)
        assert_distributions_close(dist_sym, dist_exp)

        # After observing the best shortcut closed, distributions still match.
        K = prior.condition(frozenset(), "u0", 0)
        dist_sym_K = symbolic.oracle_cost_distribution(K)

        engine = Stage0RegretSolver(T, dfa)
        idx = T.state_index["u0"]
        mass = sum(p for w, p in ep.items() if w[idx] == 0)
        dist_exp_K = defaultdict(float)
        for w, p in ep.items():
            if w[idx] == 0:
                dist_exp_K[engine.oracle_cost(w)] += p / mass
        assert_distributions_close(dist_sym_K, dict(dist_exp_K))

        assert symbolic.symbolic_nodes < len(ep)
        out[label] = {
            "explicit_worlds": len(ep),
            "symbolic_nodes": symbolic.symbolic_nodes,
            "collapsed_nodes": symbolic.collapsed_nodes,
            "oracle_shortest_path_calls": symbolic.oracle_calls,
            "oracle_support_values": len(dist_sym),
        }

    return out


def test_factored_solver_matches_explicit_small():
    dfa = eventually_goal_dfa()
    results = {}

    for label, prior_fn in [
        ("independent", lambda T: independent_prior(T, 0.30)),
        ("correlated", correlated_prior),
    ]:
        T = make_multishortcut(6)
        fp = prior_fn(T)
        explicit_prior = fp.explicit_prior_for_validation()

        for alpha in (0.0, 0.5, 0.8):
            fact = FactoredLazyHorizonDynamicCVaRSolver(
                T, dfa, fp, alpha=alpha, horizon=5
            )
            fr = fact.solve()

            ref = LazyHorizonDynamicCVaRSolver(
                T, dfa, explicit_prior, alpha=alpha, horizon=5
            )
            rr = ref.solve()

            fact_action = fr.policy[(fact.start, 5)].target
            ref_action = rr.policy[(ref.start, 5)].target

            assert fact_action == ref_action
            assert isclose(
                fr.dynamic_regret_value,
                rr.dynamic_regret_value,
                rel_tol=1e-8,
                abs_tol=1e-8,
            )

            results[(label, alpha)] = {
                "policy": fact_action,
                "factored_value": fr.dynamic_regret_value,
                "explicit_value": rr.dynamic_regret_value,
                "conceptual_worlds": fr.conceptual_world_upper_bound,
                "factored_agent_states": fr.generated_agent_states,
                "symbolic_oracle_nodes": fr.symbolic_oracle_nodes,
            }

    return results


if __name__ == "__main__":
    a = test_correlated_posterior_without_world_enumeration()
    b = test_zero_mass_assignments_are_omitted()
    c = test_symbolic_oracle_matches_explicit()
    d = test_factored_solver_matches_explicit_small()

    print("STAGE 1.6 CORE TESTS: ALL PASSED")
    print("\n[Correlated posterior]")
    for k, v in a.items():
        print(f"{k}: {v}")

    print("\n[Zero-mass assignments]")
    for k, v in b.items():
        print(f"{k}: {v}")

    print("\n[Symbolic oracle]")
    for k, v in c.items():
        print(k, v)

    print("\n[Factored solver vs explicit]")
    for k, v in d.items():
        print(k, v)
