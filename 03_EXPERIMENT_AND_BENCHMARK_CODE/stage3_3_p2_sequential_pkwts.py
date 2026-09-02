
from __future__ import annotations

from itertools import product
from typing import Dict, Tuple

from stage0_solver import DFA, PKWTS


def eventually_goal_dfa() -> DFA:
    def tr(q, label):
        if q == "qF":
            return "qF"
        return "qF" if "goal" in label else "qN"
    return DFA(("qN","qF"), "qN", frozenset({"qF"}), tr)


def sequential_revelation_pkwts():
    """
    Four-world fixed-topology PK-WTS engineered to realize a genuine
    static-CVaR continuation reversal inside the paper's own model.

    Prior:
      w1,w2 rare branch H: 0.025 each
      w3,w4 common branch L: 0.475 each

    At state y, the observed successor pattern reveals only whether the true
    world lies in {w1,w2} or {w3,w4}. In the rare branch H, a second hidden
    topology distinction remains. Two robust continuations A and B induce
    regret pairs (20,3) and (16,17), respectively. The common branch has a
    single robust continuation with regret pair (6,9).

    Therefore at alpha=.75:
      static precommitment: A is preferred at mission start;
      conditional static re-optimization after H: B is preferred;
      nested dynamic CVaR: B is prescribed at H from the outset.

    World-specific "oracle-only" passages o1..o4 are individually useful but
    not robust over the posterior branch, allowing both robust routes to have
    positive hindsight regret relative to the clairvoyant oracle.
    """
    states = (
        "s","y","h","l",
        "a","a1","a2",
        "b","b1","b2",
        "c","c3","c4",
        "o1","o2","o3","o4",
        "dead","g",
    )

    labels = {x:frozenset() for x in states}
    labels["g"] = frozenset({"goal"})

    patterns = {}
    weights = {}

    # Known prefix.
    patterns["s"] = (frozenset({"y"}),)
    weights[("s","y")] = 1.0

    # First observation: rare vs common branch.
    patterns["y"] = (
        frozenset({"h"}),  # index 0: rare worlds w1,w2
        frozenset({"l"}),  # index 1: common worlds w3,w4
    )
    weights[("y","h")] = 1.0
    weights[("y","l")] = 1.0

    # Robust choices at rare branch H.
    patterns["h"] = (frozenset({"a","b","o1","o2"}),)
    for x in ("a","b"):
        weights[("h",x)] = 1.0
    weights[("h","o1")] = 0.4
    weights[("h","o2")] = 0.4

    # Route A: total continuation cost from h is 21 in w1, 4 in w2.
    patterns["a"] = (
        frozenset({"a1"}),
        frozenset({"a2"}),
    )
    weights[("a","a1")] = 1.0
    weights[("a","a2")] = 1.0
    patterns["a1"] = (frozenset({"g"}),)
    patterns["a2"] = (frozenset({"g"}),)
    weights[("a1","g")] = 19.0
    weights[("a2","g")] = 2.0

    # Route B: total continuation cost from h is 17 in w1, 18 in w2.
    patterns["b"] = (
        frozenset({"b1"}),
        frozenset({"b2"}),
    )
    weights[("b","b1")] = 1.0
    weights[("b","b2")] = 1.0
    patterns["b1"] = (frozenset({"g"}),)
    patterns["b2"] = (frozenset({"g"}),)
    weights[("b1","g")] = 15.0
    weights[("b2","g")] = 16.0

    # Clairvoyant-only rare-branch shortcuts; each is open in only one rare world.
    patterns["o1"] = (
        frozenset({"g"}),     # open
        frozenset({"dead"}),  # closed
    )
    patterns["o2"] = (
        frozenset({"dead"}),
        frozenset({"g"}),
    )
    weights[("o1","g")] = 0.6
    weights[("o1","dead")] = 0.6
    weights[("o2","g")] = 0.6
    weights[("o2","dead")] = 0.6

    # Common branch has one robust route c and two world-specific oracle shortcuts.
    patterns["l"] = (frozenset({"c","o3","o4"}),)
    weights[("l","c")] = 1.0
    weights[("l","o3")] = 0.4
    weights[("l","o4")] = 0.4

    patterns["c"] = (
        frozenset({"c3"}),
        frozenset({"c4"}),
    )
    weights[("c","c3")] = 1.0
    weights[("c","c4")] = 1.0
    patterns["c3"] = (frozenset({"g"}),)
    patterns["c4"] = (frozenset({"g"}),)
    weights[("c3","g")] = 5.0
    weights[("c4","g")] = 8.0

    patterns["o3"] = (
        frozenset({"g"}),
        frozenset({"dead"}),
    )
    patterns["o4"] = (
        frozenset({"dead"}),
        frozenset({"g"}),
    )
    weights[("o3","g")] = 0.6
    weights[("o3","dead")] = 0.6
    weights[("o4","g")] = 0.6
    weights[("o4","dead")] = 0.6

    patterns["dead"] = (frozenset(),)
    patterns["g"] = (frozenset(),)

    # Fill patterns for every state already done; build world tuples by state order.
    idx = {x:i for i,x in enumerate(states)}
    def world(**assign):
        vals = [0] * len(states)
        for x,pidx in assign.items():
            vals[idx[x]] = pidx
        return tuple(vals)

    # Correlations enforce exactly four admissible fixed worlds.
    w1 = world(y=0, a=0, b=0, o1=0, o2=0, c=0, o3=1, o4=1)
    w2 = world(y=0, a=1, b=1, o1=1, o2=1, c=0, o3=1, o4=1)
    w3 = world(y=1, a=0, b=0, o1=1, o2=1, c=0, o3=0, o4=0)
    w4 = world(y=1, a=0, b=0, o1=1, o2=1, c=1, o3=1, o4=1)

    T = PKWTS(
        states=states,
        x0="s",
        patterns=patterns,
        weights=weights,
        labels=labels,
        allowed_worlds=(w1,w2,w3,w4),
    )
    prior = {w1:.025, w2:.025, w3:.475, w4:.475}
    names = {w1:"w1",w2:"w2",w3:"w3",w4:"w4"}
    return T, prior, names
