from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from math import isclose
from pathlib import Path
from statistics import median
import sys
from typing import Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
for import_dir in (EXPERIMENT_DIR, CORE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)


from stage1_7_utils import eventually_goal_dfa as hub_goal_dfa
from stage1_7_utils import random_multishortcut
from stage3_2_benchmark_families import distributed_layered_pkwts
from stage3_2_benchmark_families import eventually_goal_dfa as layered_goal_dfa
from stage3_2_benchmark_families import unknown_count


DATA_DIR = PROJECT_ROOT / "06_DATA_AND_RESULTS"
FIGURE_DIR = PROJECT_ROOT / "01_FIGURES"
RAW_ARTIFACT_DIR = DATA_DIR
RAW_RUNTIME_PATH = DATA_DIR / "stage3_2_controlled_benchmark_raw.csv"
FROZEN_SUMMARY_PATH = DATA_DIR / "stage3_2_controlled_benchmark_summary.csv"
RAW_WORKLOAD_PATH = DATA_DIR / "stage3_3_p1_posterior_workload_raw.csv"
FROZEN_WORKLOAD_SUMMARY_PATH = (
    DATA_DIR / "stage3_3_p1_posterior_workload_summary.csv"
)
OUTPUT_FILENAME = "stage3_5_controlled_timing_statistics.csv"
FIGURE_FILENAME = "fig_stage3_5_controlled_timing_statistics.png"
FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BenchmarkConfig:
    family: str
    m: int
    horizon: int
    seeds: tuple[int, ...]
    repetitions: int
    aggregate_role: str
    layers: int | None = None
    width: int | None = None


@dataclass(frozen=True)
class MethodSeedRecord:
    seed: int
    solve_median_s: float
    oracle_calls: int
    agent_states: int
    value_states: int


CONFIGS = (
    BenchmarkConfig("hub", 8, 3, tuple(range(5)), 3, "paired_five_seed"),
    BenchmarkConfig("hub", 10, 3, tuple(range(5)), 3, "paired_five_seed"),
    BenchmarkConfig(
        "layered", 8, 6, tuple(range(5)), 3, "paired_five_seed", 5, 2
    ),
    BenchmarkConfig(
        "layered", 10, 7, tuple(range(5)), 3, "paired_five_seed", 6, 2
    ),
    BenchmarkConfig(
        "layered", 12, 8, (0,), 1, "single_seed_stress_excluded", 7, 2
    ),
)


OUTPUT_FIELDS = (
    "family",
    "m",
    "worlds",
    "physical_states",
    "dfa_states",
    "horizon",
    "max_successor_branching",
    "max_observation_patterns",
    "aggregate_role",
    "n_seeds",
    "seed_ids",
    "warmup_runs_per_seed",
    "timed_repetitions_per_seed",
    "per_seed_runtime_statistic",
    "factored_median_s",
    "factored_q1_s",
    "factored_q3_s",
    "factored_min_s",
    "factored_max_s",
    "explicit_median_s",
    "explicit_q1_s",
    "explicit_q3_s",
    "explicit_min_s",
    "explicit_max_s",
    "paired_speedup_median_x",
    "paired_speedup_q1_x",
    "paired_speedup_q3_x",
    "paired_speedup_min_x",
    "paired_speedup_max_x",
    "paired_speedup_bootstrap_ci95_low_x",
    "paired_speedup_bootstrap_ci95_high_x",
    "bootstrap_method",
    "bootstrap_resamples",
    "single_seed_observed_ratio_x",
    "factored_oracle_calls_median",
    "factored_oracle_calls_min",
    "factored_oracle_calls_max",
    "explicit_oracle_calls_median",
    "explicit_oracle_calls_min",
    "explicit_oracle_calls_max",
    "oracle_call_reduction_median_x",
    "oracle_call_reduction_min_x",
    "oracle_call_reduction_max_x",
    "explicit_posterior_entries_median",
    "explicit_posterior_entries_min",
    "explicit_posterior_entries_max",
    "factored_probability_entries_median",
    "factored_probability_entries_min",
    "factored_probability_entries_max",
    "posterior_entry_reduction_median_x",
    "posterior_entry_reduction_min_x",
    "posterior_entry_reduction_max_x",
    "short_runtime_structural_primary",
)


def read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("A percentile requires at least one value.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must lie in [0, 1].")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("A distribution summary requires at least one value.")
    numeric = tuple(float(value) for value in values)
    return {
        "median": float(median(numeric)),
        "q1": percentile(numeric, 0.25),
        "q3": percentile(numeric, 0.75),
        "min": min(numeric),
        "max": max(numeric),
    }


def exact_bootstrap_median_ci95(
    values: Sequence[float],
) -> tuple[float, float, int]:
    numeric = tuple(float(value) for value in values)
    if len(numeric) < 2:
        raise ValueError("A bootstrap interval requires at least two values.")
    bootstrap_medians = [
        float(median(numeric[index] for index in sample_indices))
        for sample_indices in product(range(len(numeric)), repeat=len(numeric))
    ]
    return (
        percentile(bootstrap_medians, 0.025),
        percentile(bootstrap_medians, 0.975),
        len(bootstrap_medians),
    )


def format_number(value: float | int | None) -> str | int:
    if value is None:
        return ""
    if isinstance(value, int):
        return value
    return format(float(value), ".12g")


def config_key(config: BenchmarkConfig) -> tuple[str, int]:
    return config.family, config.m


def load_five_seed_runtime_records(
    config: BenchmarkConfig,
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, Dict[int, MethodSeedRecord]]:
    records: Dict[str, Dict[int, MethodSeedRecord]] = {
        "factored": {},
        "explicit": {},
    }
    for method in records:
        method_rows = [
            row
            for row in rows
            if row["family"] == config.family
            and int(row["m"]) == config.m
            and row["method"] == method
        ]
        expected_rows = len(config.seeds) * config.repetitions
        if len(method_rows) != expected_rows:
            raise RuntimeError(
                f"Expected {expected_rows} rows for {config_key(config)}/{method}, "
                f"found {len(method_rows)}."
            )
        for seed in config.seeds:
            seed_rows = [row for row in method_rows if int(row["seed"]) == seed]
            observed_repetitions = sorted(int(row["rep"]) for row in seed_rows)
            if observed_repetitions != list(range(config.repetitions)):
                raise RuntimeError(
                    f"Unexpected repetitions for {config_key(config)}/{method}/"
                    f"seed {seed}: {observed_repetitions}."
                )
            for field in (
                "worlds",
                "agent_states",
                "value_states",
                "oracle_calls",
            ):
                if len({row[field] for row in seed_rows}) != 1:
                    raise RuntimeError(
                        f"Non-deterministic {field} for {config_key(config)}/"
                        f"{method}/seed {seed}."
                    )
            if {int(row["worlds"]) for row in seed_rows} != {2**config.m}:
                raise RuntimeError(f"World count mismatch for {config_key(config)}.")
            records[method][seed] = MethodSeedRecord(
                seed=seed,
                solve_median_s=float(
                    median(float(row["solve_s"]) for row in seed_rows)
                ),
                oracle_calls=int(seed_rows[0]["oracle_calls"]),
                agent_states=int(seed_rows[0]["agent_states"]),
                value_states=int(seed_rows[0]["value_states"]),
            )
    return records


def load_single_seed_runtime_records(
    config: BenchmarkConfig,
) -> Dict[str, Dict[int, MethodSeedRecord]]:
    filenames = {
        "factored": "stage3_2_layer_m12_factored_clean.json",
        "explicit": "stage3_2_layer_m12_explicit_seed0_clean.json",
    }
    records: Dict[str, Dict[int, MethodSeedRecord]] = {}
    for method, filename in filenames.items():
        path = RAW_ARTIFACT_DIR / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(payload["H"]) != config.horizon
            or int(payload["layers"]) != config.layers
            or int(payload["width"]) != config.width
            or int(payload["reps"]) != config.repetitions
        ):
            raise RuntimeError(f"Configuration mismatch in {path}.")
        result_rows = [
            result for result in payload["results"] if int(result["seed"]) == 0
        ]
        if len(result_rows) != 1:
            raise RuntimeError(f"Expected exactly one seed-0 record in {path}.")
        result = result_rows[0]
        if int(result["m"]) != config.m:
            raise RuntimeError(f"Unknown-variable mismatch in {path}.")
        records[method] = {
            0: MethodSeedRecord(
                seed=0,
                solve_median_s=float(result["solve_median_s"]),
                oracle_calls=int(result["oracle_calls"]),
                agent_states=int(result["agent_states"]),
                value_states=int(result["value_states"]),
            )
        }
    return records


def build_transition_system(config: BenchmarkConfig, seed: int):
    if config.family == "hub":
        return random_multishortcut(config.m, seed), hub_goal_dfa()
    if config.layers is None or config.width is None:
        raise RuntimeError("Layered dimensions are missing.")
    return (
        distributed_layered_pkwts(config.layers, config.width, seed),
        layered_goal_dfa(),
    )


def structural_dimensions(config: BenchmarkConfig) -> Dict[str, int]:
    state_counts = set()
    dfa_counts = set()
    max_successor_branching = 0
    max_observation_patterns = 0
    for seed in config.seeds:
        transition_system, dfa = build_transition_system(config, seed)
        state_counts.add(len(transition_system.states))
        dfa_counts.add(len(dfa.states))
        if unknown_count(transition_system) != config.m:
            raise RuntimeError(f"Unknown-count mismatch for {config_key(config)}.")
        max_successor_branching = max(
            max_successor_branching,
            max(
                len(pattern)
                for patterns in transition_system.patterns.values()
                for pattern in patterns
            ),
        )
        max_observation_patterns = max(
            max_observation_patterns,
            max(len(patterns) for patterns in transition_system.patterns.values()),
        )
    if len(state_counts) != 1 or len(dfa_counts) != 1:
        raise RuntimeError(f"State dimensions vary for {config_key(config)}.")
    return {
        "physical_states": state_counts.pop(),
        "dfa_states": dfa_counts.pop(),
        "max_successor_branching": max_successor_branching,
        "max_observation_patterns": max_observation_patterns,
    }


def per_seed_workload(
    config: BenchmarkConfig,
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, Sequence[float]] | None:
    selected = [
        row
        for row in rows
        if row["family"] == config.family and int(row["m"]) == config.m
    ]
    if config.aggregate_role == "single_seed_stress_excluded":
        if selected:
            raise RuntimeError("The single-seed stress point unexpectedly has workload rows.")
        return None
    observed_seeds = sorted(int(row["seed"]) for row in selected)
    if observed_seeds != list(config.seeds):
        raise RuntimeError(
            f"Workload seeds mismatch for {config_key(config)}: {observed_seeds}."
        )
    explicit_entries = [
        float(row["explicit_posterior_mass_entries"]) for row in selected
    ]
    factored_entries = [
        float(row["factored_probability_entries"]) for row in selected
    ]
    reductions = [
        explicit / factored
        for explicit, factored in zip(explicit_entries, factored_entries)
    ]
    for row, reduction in zip(selected, reductions):
        if not isclose(
            reduction,
            float(row["posterior_entry_reduction_x"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise RuntimeError(f"Posterior reduction mismatch in {row}.")
    return {
        "explicit": explicit_entries,
        "factored": factored_entries,
        "reduction": reductions,
    }


def validate_frozen_summaries(
    computed_rows: Sequence[Mapping[str, object]],
) -> None:
    frozen_runtime = {
        (row["family"], int(row["m"])): row
        for row in read_csv(FROZEN_SUMMARY_PATH)
    }
    frozen_workload = {
        (row["family"], int(row["m"])): row
        for row in read_csv(FROZEN_WORKLOAD_SUMMARY_PATH)
    }
    for row in computed_rows:
        key = (str(row["family"]), int(row["m"]))
        baseline = frozen_runtime[key]
        for current_field, frozen_field in (
            ("factored_median_s", "factored_median_s"),
            ("explicit_median_s", "explicit_median_s"),
        ):
            if not isclose(
                float(row[current_field]),
                float(baseline[frozen_field]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                raise RuntimeError(f"Frozen timing mismatch for {key}/{current_field}.")
        legacy_ratio = float(row["explicit_median_s"]) / float(
            row["factored_median_s"]
        )
        if not isclose(
            legacy_ratio,
            float(baseline["speedup_x"]),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            raise RuntimeError(f"Frozen legacy ratio mismatch for {key}.")
        if int(row["n_seeds"]) != int(baseline["seeds"]) or int(
            row["timed_repetitions_per_seed"]
        ) != int(baseline["reps"]):
            raise RuntimeError(f"Frozen sample-count mismatch for {key}.")
        if key not in frozen_workload:
            continue
        workload = frozen_workload[key]
        for current_field, frozen_field in (
            (
                "explicit_posterior_entries_median",
                "explicit_posterior_mass_entries_median",
            ),
            (
                "factored_probability_entries_median",
                "factored_probability_entries_median",
            ),
            (
                "posterior_entry_reduction_median_x",
                "posterior_entry_reduction_x_median",
            ),
        ):
            if not isclose(
                float(row[current_field]),
                float(workload[frozen_field]),
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                raise RuntimeError(f"Frozen workload mismatch for {key}/{current_field}.")


def compute_rows() -> list[Dict[str, object]]:
    runtime_rows = read_csv(RAW_RUNTIME_PATH)
    if len(runtime_rows) != 120:
        raise RuntimeError(f"Expected 120 raw runtime rows, found {len(runtime_rows)}.")
    workload_rows = read_csv(RAW_WORKLOAD_PATH)
    computed_rows: list[Dict[str, object]] = []
    for config in CONFIGS:
        if config.aggregate_role == "paired_five_seed":
            records = load_five_seed_runtime_records(config, runtime_rows)
        else:
            records = load_single_seed_runtime_records(config)
        factored_times = [
            records["factored"][seed].solve_median_s for seed in config.seeds
        ]
        explicit_times = [
            records["explicit"][seed].solve_median_s for seed in config.seeds
        ]
        paired_speedups = [
            explicit / factored
            for explicit, factored in zip(explicit_times, factored_times)
        ]
        factored_oracle_calls = [
            float(records["factored"][seed].oracle_calls) for seed in config.seeds
        ]
        explicit_oracle_calls = [
            float(records["explicit"][seed].oracle_calls) for seed in config.seeds
        ]
        oracle_reductions = [
            explicit / factored
            for explicit, factored in zip(
                explicit_oracle_calls, factored_oracle_calls
            )
        ]
        factored_summary = distribution_summary(factored_times)
        explicit_summary = distribution_summary(explicit_times)
        speedup_summary = distribution_summary(paired_speedups)
        factored_oracle_summary = distribution_summary(factored_oracle_calls)
        explicit_oracle_summary = distribution_summary(explicit_oracle_calls)
        oracle_reduction_summary = distribution_summary(oracle_reductions)
        dimensions = structural_dimensions(config)
        workload = per_seed_workload(config, workload_rows)
        if config.aggregate_role == "paired_five_seed":
            bootstrap_low, bootstrap_high, bootstrap_resamples = (
                exact_bootstrap_median_ci95(paired_speedups)
            )
            bootstrap_method = (
                "exact_nonparametric_percentile_median_ordered_resamples"
            )
            single_seed_ratio = None
        else:
            bootstrap_low = None
            bootstrap_high = None
            bootstrap_resamples = 0
            bootstrap_method = "not_applicable_single_seed"
            single_seed_ratio = paired_speedups[0]
        row: Dict[str, object] = {
            "family": config.family,
            "m": config.m,
            "worlds": 2**config.m,
            **dimensions,
            "horizon": config.horizon,
            "aggregate_role": config.aggregate_role,
            "n_seeds": len(config.seeds),
            "seed_ids": ";".join(str(seed) for seed in config.seeds),
            "warmup_runs_per_seed": 1,
            "timed_repetitions_per_seed": config.repetitions,
            "per_seed_runtime_statistic": "median",
            "factored_median_s": factored_summary["median"],
            "factored_q1_s": factored_summary["q1"],
            "factored_q3_s": factored_summary["q3"],
            "factored_min_s": factored_summary["min"],
            "factored_max_s": factored_summary["max"],
            "explicit_median_s": explicit_summary["median"],
            "explicit_q1_s": explicit_summary["q1"],
            "explicit_q3_s": explicit_summary["q3"],
            "explicit_min_s": explicit_summary["min"],
            "explicit_max_s": explicit_summary["max"],
            "paired_speedup_median_x": (
                speedup_summary["median"]
                if config.aggregate_role == "paired_five_seed"
                else None
            ),
            "paired_speedup_q1_x": (
                speedup_summary["q1"]
                if config.aggregate_role == "paired_five_seed"
                else None
            ),
            "paired_speedup_q3_x": (
                speedup_summary["q3"]
                if config.aggregate_role == "paired_five_seed"
                else None
            ),
            "paired_speedup_min_x": (
                speedup_summary["min"]
                if config.aggregate_role == "paired_five_seed"
                else None
            ),
            "paired_speedup_max_x": (
                speedup_summary["max"]
                if config.aggregate_role == "paired_five_seed"
                else None
            ),
            "paired_speedup_bootstrap_ci95_low_x": bootstrap_low,
            "paired_speedup_bootstrap_ci95_high_x": bootstrap_high,
            "bootstrap_method": bootstrap_method,
            "bootstrap_resamples": bootstrap_resamples,
            "single_seed_observed_ratio_x": single_seed_ratio,
            "factored_oracle_calls_median": factored_oracle_summary["median"],
            "factored_oracle_calls_min": factored_oracle_summary["min"],
            "factored_oracle_calls_max": factored_oracle_summary["max"],
            "explicit_oracle_calls_median": explicit_oracle_summary["median"],
            "explicit_oracle_calls_min": explicit_oracle_summary["min"],
            "explicit_oracle_calls_max": explicit_oracle_summary["max"],
            "oracle_call_reduction_median_x": oracle_reduction_summary["median"],
            "oracle_call_reduction_min_x": oracle_reduction_summary["min"],
            "oracle_call_reduction_max_x": oracle_reduction_summary["max"],
            "short_runtime_structural_primary": int(
                factored_summary["median"] < 0.02
            ),
        }
        if workload is None:
            for field in (
                "explicit_posterior_entries_median",
                "explicit_posterior_entries_min",
                "explicit_posterior_entries_max",
                "factored_probability_entries_median",
                "factored_probability_entries_min",
                "factored_probability_entries_max",
                "posterior_entry_reduction_median_x",
                "posterior_entry_reduction_min_x",
                "posterior_entry_reduction_max_x",
            ):
                row[field] = None
        else:
            explicit_workload = distribution_summary(workload["explicit"])
            factored_workload = distribution_summary(workload["factored"])
            reduction_workload = distribution_summary(workload["reduction"])
            row.update(
                {
                    "explicit_posterior_entries_median": explicit_workload[
                        "median"
                    ],
                    "explicit_posterior_entries_min": explicit_workload["min"],
                    "explicit_posterior_entries_max": explicit_workload["max"],
                    "factored_probability_entries_median": factored_workload[
                        "median"
                    ],
                    "factored_probability_entries_min": factored_workload["min"],
                    "factored_probability_entries_max": factored_workload["max"],
                    "posterior_entry_reduction_median_x": reduction_workload[
                        "median"
                    ],
                    "posterior_entry_reduction_min_x": reduction_workload["min"],
                    "posterior_entry_reduction_max_x": reduction_workload["max"],
                }
            )
        computed_rows.append(row)
    validate_frozen_summaries(computed_rows)
    return computed_rows


def csv_rows(rows: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    formatted_rows = []
    text_fields = {
        "family",
        "aggregate_role",
        "seed_ids",
        "per_seed_runtime_statistic",
        "bootstrap_method",
    }
    integer_fields = {
        "m",
        "worlds",
        "physical_states",
        "dfa_states",
        "horizon",
        "max_successor_branching",
        "max_observation_patterns",
        "n_seeds",
        "warmup_runs_per_seed",
        "timed_repetitions_per_seed",
        "bootstrap_resamples",
        "short_runtime_structural_primary",
    }
    for row in rows:
        formatted = {}
        for field in OUTPUT_FIELDS:
            value = row[field]
            if field in text_fields:
                formatted[field] = value
            elif field in integer_fields:
                formatted[field] = int(value)
            else:
                formatted[field] = format_number(value)
        formatted_rows.append(formatted)
    return formatted_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_written_csv(
    path: Path,
    expected_rows: Sequence[Mapping[str, object]],
) -> None:
    observed = read_csv(path)
    normalized_expected = [
        {field: str(row[field]) for field in OUTPUT_FIELDS} for row in expected_rows
    ]
    if observed != normalized_expected:
        raise RuntimeError("The written CSV does not match the in-memory summary.")


def create_figure(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate_rows = [
        row for row in rows if row["aggregate_role"] == "paired_five_seed"
    ]
    stress_row = next(
        row
        for row in rows
        if row["aggregate_role"] == "single_seed_stress_excluded"
    )
    runtime_rows = read_csv(RAW_RUNTIME_PATH)
    style = {
        "hub": {"color": "#1f5a99", "marker": "o", "offset": -0.12},
        "layered": {"color": "#b05a00", "marker": "s", "offset": 0.12},
    }
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    ):
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        for family in ("hub", "layered"):
            family_rows = sorted(
                (row for row in aggregate_rows if row["family"] == family),
                key=lambda row: int(row["m"]),
            )
            family_style = style[family]
            x_values = [
                int(row["m"]) + family_style["offset"] for row in family_rows
            ]
            medians = [float(row["paired_speedup_median_x"]) for row in family_rows]
            axis.plot(
                x_values,
                medians,
                color=family_style["color"],
                marker=family_style["marker"],
                linewidth=1.3,
                markersize=5.5,
                label=f"{family.capitalize()} aggregate (n=5)",
                zorder=4,
            )
            for x_value, row in zip(x_values, family_rows):
                low = float(row["paired_speedup_bootstrap_ci95_low_x"])
                high = float(row["paired_speedup_bootstrap_ci95_high_x"])
                point = float(row["paired_speedup_median_x"])
                axis.errorbar(
                    x_value,
                    point,
                    yerr=[[point - low], [high - point]],
                    color=family_style["color"],
                    capsize=3,
                    linewidth=1.1,
                    zorder=3,
                )
                config = next(
                    config
                    for config in CONFIGS
                    if config.family == family and config.m == int(row["m"])
                )
                records = load_five_seed_runtime_records(config, runtime_rows)
                seed_ratios = [
                    records["explicit"][seed].solve_median_s
                    / records["factored"][seed].solve_median_s
                    for seed in config.seeds
                ]
                jitters = (-0.06, -0.03, 0.0, 0.03, 0.06)
                axis.scatter(
                    [x_value + jitter for jitter in jitters],
                    seed_ratios,
                    color=family_style["color"],
                    marker=family_style["marker"],
                    s=17,
                    alpha=0.38,
                    linewidths=0,
                    zorder=2,
                )
        stress_x = int(stress_row["m"]) + style["layered"]["offset"]
        stress_ratio = float(stress_row["single_seed_observed_ratio_x"])
        axis.scatter(
            [stress_x],
            [stress_ratio],
            marker="^",
            facecolors="none",
            edgecolors=style["layered"]["color"],
            linewidths=1.3,
            s=54,
            label="Layered stress point (n=1; no CI)",
            zorder=5,
        )
        axis.annotate(
            "stress only",
            (stress_x, stress_ratio),
            xytext=(-10, 9),
            textcoords="offset points",
            ha="right",
            color=style["layered"]["color"],
            fontsize=7.5,
        )
        axis.axhline(1.0, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_yscale("log")
        axis.set_xlim(7.45, 12.65)
        axis.set_ylim(1.0, 55.0)
        axis.set_xticks((8, 10, 12))
        axis.set_xlabel("Unknown topology variables $m$")
        axis.set_ylabel("Paired explicit / factored solve-time ratio")
        axis.legend(loc="upper left", frameon=False)
        axis.grid(axis="y", which="major", color="#d9d9d9", linewidth=0.6)
        axis.grid(axis="y", which="minor", visible=False)
        figure.tight_layout(pad=0.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={"Software": f"Matplotlib {matplotlib.__version__}"},
        )
        plt.close(figure)


def print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        key = f"{row['family']}/m={row['m']}"
        if row["aggregate_role"] == "paired_five_seed":
            print(
                f"[SUMMARY] {key}: paired median="
                f"{float(row['paired_speedup_median_x']):.4f}x, range="
                f"[{float(row['paired_speedup_min_x']):.4f}, "
                f"{float(row['paired_speedup_max_x']):.4f}], exact bootstrap "
                f"95% CI=[{float(row['paired_speedup_bootstrap_ci95_low_x']):.4f}, "
                f"{float(row['paired_speedup_bootstrap_ci95_high_x']):.4f}]."
            )
        else:
            print(
                f"[SUMMARY] {key}: single-seed observed ratio="
                f"{float(row['single_seed_observed_ratio_x']):.4f}x; "
                "excluded from aggregate inference."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired controlled-runtime dispersion, exact small-sample "
            "bootstrap intervals, and structural scaling evidence."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "reproduced_results" / OUTPUT_FILENAME,
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=PROJECT_ROOT / "reproduced_results" / FIGURE_FILENAME,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = compute_rows()
    formatted_rows = csv_rows(rows)
    output_csv = args.output_csv.resolve()
    output_figure = args.output_figure.resolve()
    write_csv(output_csv, formatted_rows)
    validate_written_csv(output_csv, formatted_rows)
    create_figure(output_figure, rows)
    print(f"[PASS] Wrote {len(rows)} rows to {output_csv}")
    print(f"[PASS] Wrote controlled timing figure to {output_figure}")
    print("[PASS] Raw pairing, structural dimensions, and frozen anchors match.")
    print(f"[HASH] {output_csv.name}: {sha256(output_csv)}")
    print(f"[HASH] {output_figure.name}: {sha256(output_figure)}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
