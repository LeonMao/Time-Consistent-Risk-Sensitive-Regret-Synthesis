
from __future__ import annotations
from math import hypot
from typing import Dict, Tuple

from stage0_solver import DFA, PKWTS
from factored_belief import MixtureProductPrior, ProductComponent


def firefighting_dfa() -> DFA:
    """
    scLTL: (not fire U extinguisher) AND eventually fire.

    q0: extinguisher not yet acquired
    q1: extinguisher acquired, fire not yet visited
    qF: task accepted
    qD: fire visited before extinguisher (rejecting sink)
    """
    def tr(q, label):
        if q == "qF":
            return "qF"
        if q == "qD":
            return "qD"
        has_e = "extinguisher" in label
        has_f = "fire" in label
        if q == "q0":
            if has_f and not has_e:
                return "qD"
            if has_e and has_f:
                return "qF"
            if has_e:
                return "q1"
            return "q0"
        # q1
        return "qF" if has_f else "q1"

    return DFA(
        states=("q0", "q1", "qF", "qD"),
        q_init="q0",
        accepting=frozenset({"qF"}),
        transition_fn=tr,
    )


def intel_lab_topological_pkwts() -> Tuple[PKWTS, Dict[str, Tuple[float, float]]]:
    """
    Hand-extracted topological abstraction of the public Intel Research Lab
    occupancy-map geometry. It preserves the prominent perimeter corridor and
    side-room/cross-passage structure. Coordinates are schematic map coordinates,
    used only to set metric-like edge weights and draw the benchmark.

    Five uncertain passage-probe states d0..d4 are binary monotone:
      closed -> return to the approach node
      open   -> return or traverse a shortcut
    """
    coords = {
        "s": (2.8, 0.5),       # lower corridor / start
        "bl": (1.2, 0.8),
        "lm": (0.8, 2.4),
        "lt": (1.0, 4.2),
        "tm": (3.0, 4.7),
        "tr": (5.0, 4.4),
        "rm": (5.6, 2.7),
        "br": (5.2, 0.9),

        "ext": (0.15, 2.6),    # resource room
        "fire": (6.35, 2.7),   # incident room

        # Uncertain passage probes.
        "d0": (1.8, 1.55),     # s -> lm
        "d1": (1.85, 3.7),     # lm -> tm
        "d2": (4.15, 3.75),    # tm -> rm
        "d3": (4.25, 1.55),    # s -> rm
        "d4": (3.25, 2.7),     # lt -> rm cross passage
    }

    safe_edges = [
        ("s", "bl"), ("bl", "lm"), ("lm", "lt"), ("lt", "tm"),
        ("tm", "tr"), ("tr", "rm"), ("rm", "br"), ("br", "s"),
        ("lm", "ext"), ("rm", "fire"),
    ]

    # Known graph is bidirectional on the perimeter/rooms.
    known_succ = {x: set() for x in coords}
    weights = {}

    def add_known(a, b, scale=2.0):
        d = scale * hypot(coords[a][0]-coords[b][0], coords[a][1]-coords[b][1])
        known_succ[a].add(b)
        known_succ[b].add(a)
        weights[(a,b)] = d
        weights[(b,a)] = d

    for a,b in safe_edges:
        add_known(a,b)

    shortcuts = {
        "d0": ("s", "lm"),
        "d1": ("lm", "tm"),
        "d2": ("tm", "rm"),
        "d3": ("s", "rm"),
        "d4": ("lt", "rm"),
    }

    # Approach edges are known. Door states themselves encode blocked/open outcome.
    for d, (origin, target) in shortcuts.items():
        approach = 0.55 * 2.0 * hypot(
            coords[d][0]-coords[origin][0], coords[d][1]-coords[origin][1]
        )
        traversal = 1.15 * 2.0 * hypot(
            coords[d][0]-coords[target][0], coords[d][1]-coords[target][1]
        )
        known_succ[origin].add(d)
        weights[(origin,d)] = approach
        # if blocked the robot returns to the origin
        weights[(d,origin)] = approach
        # if open the passage gives an additional traversal option
        weights[(d,target)] = traversal

    # Calibrated map-derived entrance d0: open passage saves a meaningful
    # corridor detour, while a blocked inspection incurs a small backtrack.
    # This preserves the benchmark geometry but exposes the intended
    # exploration-versus-tail-risk trade-off.
    weights[("s","d0")] = 1.20
    weights[("d0","s")] = 1.20
    weights[("d0","lm")] = 3.00

    patterns = {}
    for x in coords:
        if x in shortcuts:
            origin, target = shortcuts[x]
            patterns[x] = (
                frozenset({origin}),
                frozenset({origin, target}),
            )
        else:
            patterns[x] = (frozenset(known_succ[x]),)

    labels = {x: frozenset() for x in coords}
    labels["ext"] = frozenset({"extinguisher"})
    labels["fire"] = frozenset({"fire"})

    T = PKWTS(
        states=tuple(coords.keys()),
        x0="s",
        patterns=patterns,
        weights=weights,
        labels=labels,
    )
    return T, coords


def intel_two_mode_prior(T: PKWTS) -> MixtureProductPrior:
    """
    Correlated building-access prior:
    normal-operation mode: passages are usually open
    restricted-access mode: passages are usually closed
    """
    good = {}
    restricted = {}
    for x in T.states:
        if len(T.patterns[x]) > 1:
            good[x] = (0.22, 0.78)
            restricted[x] = (0.82, 0.18)
    return MixtureProductPrior(
        T,
        [
            ProductComponent(0.55, good),
            ProductComponent(0.45, restricted),
        ],
    )
