from __future__ import annotations

import argparse
import csv
import json
from math import hypot, isclose
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for import_dir in (CORE_DIR, EXPERIMENT_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from factored_belief import MixtureProductPrior, ProductComponent
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from lazy_dynamic_cvar_solver import LazyHorizonDynamicCVaRSolver
from stage0_solver import DFA, PKWTS, World
from stage3_3_intel_lab_benchmark import (
    firefighting_dfa,
    intel_lab_topological_pkwts,
    intel_two_mode_prior,
)
from stage3_5_prior_robustness import (
    evaluate_fixed_policy,
    simulate_fixed_policy_world,
)


DEFAULT_JSON_PATH = (
    PROJECT_ROOT / "reproduced_results" / "stage3_5_intel_spec.json"
)
DEFAULT_TEX_PATH = (
    PROJECT_ROOT / "reproduced_results" / "stage3_5_intel_spec_table.tex"
)
REFERENCE_RESULTS_PATH = (
    PROJECT_ROOT / "06_DATA_AND_RESULTS" / "stage3_3_intel_lab_results.csv"
)
NOMINAL_HORIZON = 11
MINIMAL_ROBUST_RANK = 9
PLANNING_ALPHAS = (0.0, 0.25, 0.50, 0.75, 0.90)
TERMINAL_EVALUATION_ALPHA = 0.95
VALUE_TOLERANCE = 1e-10


def ordered_states(states: Sequence[str], values: Iterable[str]) -> list[str]:
    index = {state: position for position, state in enumerate(states)}
    return sorted(values, key=index.__getitem__)


def infer_passages(
    transition_system: PKWTS,
) -> Dict[str, Dict[str, object]]:
    passages: Dict[str, Dict[str, object]] = {}
    for variable in transition_system.states:
        patterns = transition_system.patterns[variable]
        if len(patterns) <= 1:
            continue
        if len(patterns) != 2:
            raise ValueError(f"Intel variable {variable} is not binary.")
        closed, opened = patterns
        common = closed & opened
        added = opened - closed
        if len(common) != 1 or len(added) != 1 or not closed < opened:
            raise ValueError(
                f"Intel variable {variable} is not a monotone passage pattern."
            )
        passages[variable] = {
            "origin": next(iter(common)),
            "open_target": next(iter(added)),
            "closed_pattern_index": 0,
            "open_pattern_index": 1,
        }
    return passages


def edge_class(
    source: str,
    target: str,
    passages: Mapping[str, Mapping[str, object]],
) -> str:
    if source in passages:
        passage = passages[source]
        if target == passage["origin"]:
            return "closed_return"
        if target == passage["open_target"]:
            return "open_only_shortcut"
    if target in passages and source == passages[target]["origin"]:
        return "known_probe_approach"
    return "known_corridor_or_room"


def expected_formula_cost(
    source: str,
    target: str,
    classification: str,
    coordinates: Mapping[str, Tuple[float, float]],
) -> Tuple[float, str]:
    distance = hypot(
        coordinates[source][0] - coordinates[target][0],
        coordinates[source][1] - coordinates[target][1],
    )
    if classification == "known_corridor_or_room":
        return 2.0 * distance, "2.0 * euclidean_coordinate_distance"
    if classification in {"known_probe_approach", "closed_return"}:
        return 1.1 * distance, "1.1 * euclidean_coordinate_distance"
    if classification == "open_only_shortcut":
        return 2.3 * distance, "2.3 * euclidean_coordinate_distance"
    raise ValueError(f"Unknown edge classification: {classification}")


def reference_results() -> list[Dict[str, object]]:
    with REFERENCE_RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        selected.append(
            {
                "planning_alpha": float(row["alpha"]),
                "horizon": int(row["H"]),
                "supported_worlds": int(row["worlds"]),
                "dfa_states": int(row["dfa_states"]),
                "root_action": row["root_action"],
                "nested_dynamic_regret": float(row["dynamic_regret_objective"]),
                "exact_mean_cost": float(row["exact_mean_cost"]),
                "exact_mean_regret": float(row["exact_mean_regret"]),
                "exact_static_cvar95_regret": float(row["exact_cvar95_regret"]),
                "exact_worst_regret": float(row["exact_worst_regret"]),
                "all_worlds_satisfied": bool(int(row["all_worlds_satisfied"])),
            }
        )
    return selected


def dfa_label_classes() -> Tuple[Tuple[str, frozenset[str]], ...]:
    return (
        ("none", frozenset()),
        ("extinguisher", frozenset({"extinguisher"})),
        ("fire", frozenset({"fire"})),
        ("extinguisher_and_fire", frozenset({"extinguisher", "fire"})),
    )


def build_specification() -> Dict[str, object]:
    transition_system, coordinates = intel_lab_topological_pkwts()
    dfa = firefighting_dfa()
    prior = intel_two_mode_prior(transition_system)
    passages = infer_passages(transition_system)
    state_index = transition_system.state_index

    nodes = []
    for state in transition_system.states:
        nodes.append(
            {
                "name": state,
                "coordinate": [
                    float(coordinates[state][0]),
                    float(coordinates[state][1]),
                ],
                "atomic_propositions": sorted(transition_system.labels[state]),
            }
        )

    edges = []
    for source, target in sorted(
        transition_system.weights,
        key=lambda edge: (state_index[edge[0]], state_index[edge[1]]),
    ):
        classification = edge_class(source, target, passages)
        expected_cost, formula = expected_formula_cost(
            source,
            target,
            classification,
            coordinates,
        )
        actual_cost = float(transition_system.weights[(source, target)])
        edges.append(
            {
                "source": source,
                "target": target,
                "edge_class": classification,
                "cost": actual_cost,
                "default_formula": formula,
                "default_formula_cost": expected_cost,
                "manual_override": not isclose(
                    actual_cost,
                    expected_cost,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            }
        )

    known_pairs = []
    seen_pairs = set()
    for edge in edges:
        if edge["edge_class"] != "known_corridor_or_room":
            continue
        source = str(edge["source"])
        target = str(edge["target"])
        canonical = tuple(sorted((source, target), key=state_index.__getitem__))
        if canonical not in seen_pairs:
            seen_pairs.add(canonical)
            known_pairs.append(list(canonical))

    successor_patterns = []
    for state in transition_system.states:
        successor_patterns.append(
            {
                "state": state,
                "patterns": [
                    ordered_states(transition_system.states, pattern)
                    for pattern in transition_system.patterns[state]
                ],
            }
        )

    uncertain_variables = []
    for variable in prior.variables:
        passage = passages[variable]
        uncertain_variables.append(
            {
                "name": variable,
                "origin": passage["origin"],
                "open_target": passage["open_target"],
                "pattern_semantics": [
                    {
                        "index": 0,
                        "name": "closed",
                        "successors": ordered_states(
                            transition_system.states,
                            transition_system.patterns[variable][0],
                        ),
                    },
                    {
                        "index": 1,
                        "name": "open",
                        "successors": ordered_states(
                            transition_system.states,
                            transition_system.patterns[variable][1],
                        ),
                    },
                ],
            }
        )

    component_names = ("normal_access", "restricted_access")
    components = []
    for name, component in zip(component_names, prior.components):
        components.append(
            {
                "name": name,
                "mixture_weight": float(component.weight),
                "variable_probabilities": {
                    variable: {
                        "closed_pattern_0": float(component.marginals[variable][0]),
                        "open_pattern_1": float(component.marginals[variable][1]),
                    }
                    for variable in prior.variables
                },
            }
        )

    label_classes = dfa_label_classes()
    dfa_transitions = []
    for state in dfa.states:
        for label_name, propositions in label_classes:
            dfa_transitions.append(
                {
                    "from": state,
                    "label_class": label_name,
                    "to": dfa.step(state, propositions),
                }
            )

    return {
        "schema": "stage3_5_intel_benchmark_spec_v1",
        "benchmark": {
            "name": "intel_map_scLTL",
            "description": (
                "Schematic topological abstraction derived from the public Intel "
                "Research Lab occupancy-map geometry."
            ),
            "source_builder": (
                "03_EXPERIMENT_AND_BENCHMARK_CODE/"
                "stage3_3_intel_lab_benchmark.py"
            ),
            "physical_start_state": transition_system.x0,
            "node_count": len(transition_system.states),
            "uncertain_variable_count": len(prior.variables),
            "supported_world_count": prior.conceptual_world_upper_bound(),
        },
        "coordinate_system": {
            "status": "schematic_map_coordinates",
            "physical_units": "none",
            "interpretation": (
                "Coordinates preserve topological layout and set metric-like "
                "synthetic costs; they are not map meters or measured travel times."
            ),
        },
        "cost_model": {
            "status": "synthetic_geometric_with_manual_calibration",
            "known_corridor_or_room": "2.0 * euclidean_coordinate_distance",
            "probe_approach_and_closed_return": (
                "1.1 * euclidean_coordinate_distance"
            ),
            "open_only_shortcut": "2.3 * euclidean_coordinate_distance",
            "manual_overrides": [
                {"source": "s", "target": "d0", "cost": 1.2},
                {"source": "d0", "target": "s", "cost": 1.2},
                {"source": "d0", "target": "lm", "cost": 3.0},
            ],
            "interpretation": (
                "All costs are dimensionless synthetic map-coordinate weights. "
                "The d0 calibration exposes the intended exploration/tail-risk trade-off."
            ),
        },
        "nodes": nodes,
        "known_undirected_corridor_or_room_edges": known_pairs,
        "directed_weighted_edges": edges,
        "successor_patterns": successor_patterns,
        "uncertain_variables": uncertain_variables,
        "world_encoding": {
            "variable_order": list(prior.variables),
            "closed_value": 0,
            "open_value": 1,
            "world_vector_length": len(prior.variables),
            "supported_world_count": prior.conceptual_world_upper_bound(),
        },
        "prior": {
            "type": "finite_mixture_of_product_categorical_factors",
            "formula": (
                "P(z)=0.55*prod_i(0.22^(1-z_i)*0.78^z_i)+"
                "0.45*prod_i(0.82^(1-z_i)*0.18^z_i)"
            ),
            "components": components,
        },
        "task": {
            "name": "firefighting",
            "formula_ascii": "(!fire U extinguisher) AND eventually fire",
            "formula_latex": (
                "(\\neg fire\\ \\mathsf{U}\\ extinguisher)"
                "\\wedge\\Diamond fire"
            ),
        },
        "dfa": {
            "states": list(dfa.states),
            "initial_state": dfa.q_init,
            "accepting_states": sorted(dfa.accepting),
            "rejecting_sink_states": ["qD"],
            "label_classes": [
                {"name": name, "atomic_propositions": sorted(propositions)}
                for name, propositions in label_classes
            ],
            "transition_table": dfa_transitions,
        },
        "planning_and_evaluation": {
            "nominal_horizon": NOMINAL_HORIZON,
            "minimal_robust_rank": MINIMAL_ROBUST_RANK,
            "planning_alphas": list(PLANNING_ALPHAS),
            "terminal_evaluation_alpha": TERMINAL_EVALUATION_ALPHA,
            "terminal_evaluation_statistic": "static_CVaR_of_terminal_regret",
            "world_enumeration": "exact",
        },
        "reference_results_h11": reference_results(),
    }


def label_lookup(specification: Mapping[str, object]) -> Dict[str, frozenset[str]]:
    dfa_spec = specification["dfa"]
    return {
        str(entry["name"]): frozenset(entry["atomic_propositions"])
        for entry in dfa_spec["label_classes"]
    }


def reconstruct_from_specification(
    specification: Mapping[str, object],
) -> Tuple[PKWTS, Dict[str, Tuple[float, float]], MixtureProductPrior, DFA]:
    nodes = specification["nodes"]
    states = tuple(str(node["name"]) for node in nodes)
    coordinates = {
        str(node["name"]): (
            float(node["coordinate"][0]),
            float(node["coordinate"][1]),
        )
        for node in nodes
    }
    labels = {
        str(node["name"]): frozenset(node["atomic_propositions"])
        for node in nodes
    }
    patterns = {
        str(entry["state"]): tuple(
            frozenset(pattern) for pattern in entry["patterns"]
        )
        for entry in specification["successor_patterns"]
    }
    weights = {
        (str(edge["source"]), str(edge["target"])): float(edge["cost"])
        for edge in specification["directed_weighted_edges"]
    }
    transition_system = PKWTS(
        states=states,
        x0=str(specification["benchmark"]["physical_start_state"]),
        patterns=patterns,
        weights=weights,
        labels=labels,
    )

    prior_components = []
    for component in specification["prior"]["components"]:
        marginals = {
            variable: (
                float(probabilities["closed_pattern_0"]),
                float(probabilities["open_pattern_1"]),
            )
            for variable, probabilities in component[
                "variable_probabilities"
            ].items()
        }
        prior_components.append(
            ProductComponent(
                weight=float(component["mixture_weight"]),
                marginals=marginals,
            )
        )
    prior = MixtureProductPrior(transition_system, prior_components)

    dfa_spec = specification["dfa"]
    classes = label_lookup(specification)
    class_by_label = {propositions: name for name, propositions in classes.items()}
    transitions = {
        (str(entry["from"]), str(entry["label_class"])): str(entry["to"])
        for entry in dfa_spec["transition_table"]
    }

    def transition(state: str, propositions: frozenset[str]) -> str:
        if propositions not in class_by_label:
            raise ValueError(f"Unspecified DFA label class: {sorted(propositions)}")
        return transitions[(state, class_by_label[propositions])]

    dfa = DFA(
        states=tuple(str(state) for state in dfa_spec["states"]),
        q_init=str(dfa_spec["initial_state"]),
        accepting=frozenset(str(state) for state in dfa_spec["accepting_states"]),
        transition_fn=transition,
    )
    return transition_system, coordinates, prior, dfa


def assert_close(label: str, actual: float, expected: float) -> None:
    if not isclose(actual, expected, rel_tol=0.0, abs_tol=VALUE_TOLERANCE):
        raise RuntimeError(
            f"Intel specification mismatch for {label}: "
            f"actual={actual!r}, expected={expected!r}."
        )


def assert_structural_identity(
    specification: Mapping[str, object],
    reconstructed_system: PKWTS,
    reconstructed_coordinates: Mapping[str, Tuple[float, float]],
    reconstructed_prior: MixtureProductPrior,
    reconstructed_dfa: DFA,
) -> None:
    source_system, source_coordinates = intel_lab_topological_pkwts()
    source_prior = intel_two_mode_prior(source_system)
    source_dfa = firefighting_dfa()
    if reconstructed_system.states != source_system.states:
        raise RuntimeError("Reconstructed Intel state order differs from the source.")
    if reconstructed_system.x0 != source_system.x0:
        raise RuntimeError("Reconstructed Intel start state differs from the source.")
    if reconstructed_system.patterns != source_system.patterns:
        raise RuntimeError("Reconstructed Intel successor patterns differ from the source.")
    if reconstructed_system.labels != source_system.labels:
        raise RuntimeError("Reconstructed Intel labels differ from the source.")
    if reconstructed_system.weights.keys() != source_system.weights.keys():
        raise RuntimeError("Reconstructed Intel directed edge set differs from the source.")
    for edge, expected in source_system.weights.items():
        assert_close(
            f"directed edge {edge}",
            reconstructed_system.weights[edge],
            expected,
        )
    if reconstructed_coordinates != source_coordinates:
        raise RuntimeError("Reconstructed Intel coordinates differ from the source.")
    source_explicit = source_prior.explicit_prior_for_validation()
    reconstructed_explicit = reconstructed_prior.explicit_prior_for_validation()
    if source_explicit.keys() != reconstructed_explicit.keys():
        raise RuntimeError("Reconstructed Intel prior support differs from the source.")
    for world, expected in source_explicit.items():
        assert_close(
            f"world probability {world}",
            reconstructed_explicit[world],
            expected,
        )
    if reconstructed_dfa.states != source_dfa.states:
        raise RuntimeError("Reconstructed Intel DFA states differ from the source.")
    for state in source_dfa.states:
        for _, propositions in dfa_label_classes():
            if reconstructed_dfa.step(state, propositions) != source_dfa.step(
                state,
                propositions,
            ):
                raise RuntimeError(
                    f"Reconstructed Intel DFA transition differs at {state}."
                )
    if len(reconstructed_explicit) != int(
        specification["benchmark"]["supported_world_count"]
    ):
        raise RuntimeError("Reconstructed Intel world count is incorrect.")


def validate_reference_results(
    specification: Mapping[str, object],
    transition_system: PKWTS,
    prior: MixtureProductPrior,
    dfa: DFA,
) -> None:
    explicit_prior = prior.explicit_prior_for_validation()
    for reference in specification["reference_results_h11"]:
        alpha = float(reference["planning_alpha"])
        horizon = int(reference["horizon"])
        solver = FactoredLazyHorizonDynamicCVaRSolver(
            transition_system,
            dfa,
            prior,
            alpha,
            horizon,
        )
        result = solver.solve()
        root_action = result.policy[(solver.start, horizon)].target
        if root_action != reference["root_action"]:
            raise RuntimeError(
                f"Reconstructed Intel root action differs at alpha={alpha}."
            )
        assert_close(
            f"nested objective alpha={alpha}",
            result.dynamic_regret_value,
            float(reference["nested_dynamic_regret"]),
        )

        evaluator = LazyHorizonDynamicCVaRSolver(
            transition_system,
            dfa,
            explicit_prior,
            alpha,
            horizon,
        )
        evaluation = evaluate_fixed_policy(evaluator, result.policy)
        if evaluation.status != "complete":
            raise RuntimeError(
                f"Reconstructed Intel policy is incomplete at alpha={alpha}."
            )
        assert_close(
            f"mean regret alpha={alpha}",
            float(evaluation.mean_regret),
            float(reference["exact_mean_regret"]),
        )
        assert_close(
            f"static CVaR.95 alpha={alpha}",
            float(evaluation.cvar95_regret),
            float(reference["exact_static_cvar95_regret"]),
        )
        assert_close(
            f"worst regret alpha={alpha}",
            float(evaluation.worst_regret),
            float(reference["exact_worst_regret"]),
        )
        mean_cost = 0.0
        for world, probability in explicit_prior.items():
            cost, terminal, status = simulate_fixed_policy_world(
                evaluator,
                result.policy,
                world,
            )
            if status != "accepted" or cost is None or terminal.q not in dfa.accepting:
                raise RuntimeError(
                    f"Reconstructed Intel policy failed world {world} at alpha={alpha}."
                )
            mean_cost += probability * cost
        assert_close(
            f"mean cost alpha={alpha}",
            mean_cost,
            float(reference["exact_mean_cost"]),
        )
        if evaluation.all_worlds_satisfied != bool(
            reference["all_worlds_satisfied"]
        ):
            raise RuntimeError(
                f"Reconstructed Intel satisfaction differs at alpha={alpha}."
            )
        print(
            f"[VALIDATED] alpha={alpha:.2f} root={root_action} "
            f"objective={result.dynamic_regret_value:.12f}",
            flush=True,
        )


def tex_state(name: str) -> str:
    if name.startswith("d") and name[1:].isdigit():
        return f"\\ensuremath{{d_{name[1:]}}}"
    return f"\\texttt{{{name}}}"


def tex_coordinate_rows(specification: Mapping[str, object]) -> Tuple[str, str]:
    entries = [
        f"{tex_state(str(node['name']))} $({float(node['coordinate'][0]):.2f},"
        f"{float(node['coordinate'][1]):.2f})$"
        for node in specification["nodes"]
    ]
    midpoint = 8
    return ", ".join(entries[:midpoint]), ", ".join(entries[midpoint:])


def render_latex_table(specification: Mapping[str, object]) -> str:
    coordinate_first, coordinate_second = tex_coordinate_rows(specification)
    known_edges = ", ".join(
        f"{tex_state(edge[0])}--{tex_state(edge[1])}"
        for edge in specification["known_undirected_corridor_or_room_edges"]
    )
    passages = ", ".join(
        f"{tex_state(str(variable['name']))}:"
        f"({tex_state(str(variable['origin']))},{tex_state(str(variable['open_target']))})"
        for variable in specification["uncertain_variables"]
    )
    alphas = ",".join(
        f"{float(alpha):.2f}".lstrip("0")
        for alpha in specification["planning_and_evaluation"]["planning_alphas"]
    )
    return f"""% Auto-generated by stage3_5_export_intel_spec.py; do not edit by hand.
\\begin{{table*}}[t]
\\centering
\\caption{{Self-contained Intel map-derived benchmark specification. For each passage $d_i:(o_i,t_i)$, pattern 0 is closed and pattern 1 is open. Coordinates are schematic and all costs are dimensionless synthetic weights, not meters or measured travel times.}}
\\label{{tab:intel-spec}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}
\\begin{{tabular}}{{p{{3.0cm}}p{{14.0cm}}}}
\\toprule
Item & Complete specification\\\\
\\midrule
States and coordinates & {coordinate_first}.\\\\
 & {coordinate_second}.\\\\
Start and labels & $x_0=\\texttt{{s}}$; $L(\\texttt{{ext}})=\\{{extinguisher\\}}$, $L(\\texttt{{fire}})=\\{{fire\\}}$, and all other labels are empty.\\\\
Known undirected edges & {known_edges}. Each direction has cost $2\\lVert p_u-p_v\\rVert_2$.\\\\
Uncertain passages & {passages}, where each tuple gives (origin, open target). The approach $o_i\\to d_i$ is known; $S^0(d_i)=\\{{o_i\\}}$ and $S^1(d_i)=\\{{o_i,t_i\\}}$.\\\\
Passage costs & Normally $c(o_i,d_i)=c(d_i,o_i)=1.1\\lVert p_{{o_i}}-p_{{d_i}}\\rVert_2$ and $c(d_i,t_i)=2.3\\lVert p_{{d_i}}-p_{{t_i}}\\rVert_2$. Manual calibration: $c(\\texttt{{s}},d_0)=c(d_0,\\texttt{{s}})=1.20$ and $c(d_0,\\texttt{{lm}})=3.00$.\\\\
Correlated prior & For $z_i=1$ iff $d_i$ is open, $P(z)=0.55\\prod_i(0.22)^{{1-z_i}}(0.78)^{{z_i}}+0.45\\prod_i(0.82)^{{1-z_i}}(0.18)^{{z_i}}$. Thus all $2^5=32$ worlds have positive mass.\\\\
Task and DFA & $\\varphi_{{\\rm fire}}=(\\neg fire\\ \\mathsf{{U}}\\ extinguisher)\\wedge\\Diamond fire$. States $(q_0,q_1,q_F,q_D)$, initial $q_0$, accepting $q_F$: $q_0$ maps $\\emptyset/e/f/ef$ to $q_0/q_1/q_D/q_F$; $q_1$ maps $\\emptyset,e$ to $q_1$ and $f,ef$ to $q_F$; $q_F,q_D$ self-loop. Here $e$ and $f$ denote the two propositions.\\\\
Planning/evaluation & Minimal robust rank 9; nominal $H=11$; planning $\\alpha\\in\\{{{alphas}\\}}$; terminal reporting uses static $\\operatorname{{CVaR}}_{{.95}}(R)$ and exact enumeration.\\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export and independently reconstruct the complete Intel benchmark "
            "specification."
        )
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specification = build_specification()
    json_text = json.dumps(
        specification,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    roundtrip_specification = json.loads(json_text)
    reconstructed = reconstruct_from_specification(roundtrip_specification)
    assert_structural_identity(roundtrip_specification, *reconstructed)
    validate_reference_results(roundtrip_specification, reconstructed[0], reconstructed[2], reconstructed[3])
    latex_text = render_latex_table(roundtrip_specification)

    json_path = args.json_output.resolve()
    tex_path = args.tex_output.resolve()
    atomic_write(json_path, json_text)
    atomic_write(tex_path, latex_text)
    print(f"[PASS] Wrote complete Intel JSON specification to {json_path}")
    print(f"[PASS] Wrote generated Intel LaTeX table to {tex_path}")
    print("[PASS] JSON round-trip reconstructs 15 states, 5 variables, and 32 worlds.")
    print("[PASS] Five H=11 reference result rows reproduce exactly within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
