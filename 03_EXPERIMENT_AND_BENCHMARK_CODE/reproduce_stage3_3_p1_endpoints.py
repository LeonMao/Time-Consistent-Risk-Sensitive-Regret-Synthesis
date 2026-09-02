from __future__ import annotations

import argparse
from collections import deque
import csv
from math import isclose
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CORE_DIR = PROJECT_ROOT / "02_CORE_CODE"
for import_dir in (EXPERIMENT_DIR, CORE_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from stage1_7_utils import eventually_goal_dfa, independent_prior, random_multishortcut
from stage3_3_intel_lab_benchmark import (
    firefighting_dfa,
    intel_lab_topological_pkwts,
    intel_two_mode_prior,
)
from factored_dynamic_cvar_solver import FactoredLazyHorizonDynamicCVaRSolver
from stage3_3_baseline_solvers import FactoredWorstCaseRegretSolver


HUB_GRID = (0.0, 0.25, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
            0.9, 0.925, 0.95, 0.975, 0.99)
REPRESENTATIVE_GRID = (
    0.0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def minimax(transition_system, dfa, prior, horizon: int):
    solver = FactoredWorstCaseRegretSolver(
        transition_system, dfa, prior, alpha=0.5, horizon=horizon
    )
    result = solver.solve_regret()
    return float(result.objective), str(result.policy[(solver.start, horizon)].target)


def local_atom_threshold(transition_system, dfa, prior, horizon: int) -> dict:
    solver = FactoredLazyHorizonDynamicCVaRSolver(
        transition_system, dfa, prior, alpha=0.5, horizon=horizon
    )
    pending = deque([(solver.start, horizon)])
    visited = set()
    branch_atoms: list[float] = []
    terminal_atoms: list[float] = []

    while pending:
        agent, remaining = pending.popleft()
        key = (agent, remaining)
        if key in visited:
            continue
        visited.add(key)
        if agent.q in solver.A.accepting:
            distribution = solver.oracle.oracle_cost_distribution(agent.K)
            terminal_atoms.extend(
                float(probability)
                for probability in distribution.values()
                if probability > 0.0
            )
            continue
        if remaining <= 0:
            continue
        for env in solver.actions(agent):
            children = solver.branches(env)
            if not children or not all(
                solver.can_accept_within(child, remaining - 1)
                for child in children
            ):
                continue
            branch_atoms.extend(
                float(probability)
                for probability in children.values()
                if probability > 0.0
            )
            pending.extend((child, remaining - 1) for child in children)

    if not branch_atoms or not terminal_atoms:
        raise RuntimeError("Endpoint atom scan did not reach both risk-map types.")
    branch_min = min(branch_atoms)
    terminal_min = min(terminal_atoms)
    local_min = min(branch_min, terminal_min)
    return {
        "pmin_dynamic_support": local_min,
        "branch_pmin": branch_min,
        "terminal_gap_pmin": terminal_min,
        "sufficient_alpha_threshold": 1.0 - local_min,
        "reachable_augmented_states": len(visited),
    }


def solve_grid(transition_system, dfa, prior, horizon: int, alphas):
    minimum_value, minimum_root = minimax(
        transition_system, dfa, prior, horizon
    )
    rows = []
    for alpha in alphas:
        solver = FactoredLazyHorizonDynamicCVaRSolver(
            transition_system, dfa, prior, alpha=alpha, horizon=horizon
        )
        result = solver.solve()
        value = float(result.dynamic_regret_value)
        rows.append({
            "alpha": alpha,
            "dynamic_value": value,
            "root_action": str(result.policy[(solver.start, horizon)].target),
            "minimax_value": minimum_value,
            "minimax_root": minimum_root,
            "value_gap_to_minimax": minimum_value - value,
            "at_minimax_value": int(
                isclose(value, minimum_value, rel_tol=0.0, abs_tol=1e-10)
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstructed endpoint driver for Robotica Figure 6."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dfa = eventually_goal_dfa()
    hub_plateau: list[dict] = []
    representative: list[dict] = []
    thresholds: list[dict] = []

    for m in (8, 10):
        for seed in range(10):
            transition_system = random_multishortcut(m, seed)
            prior = independent_prior(
                transition_system, 0.30, seed=seed, jitter=0.08
            )
            minimum_value, minimum_root = minimax(
                transition_system, dfa, prior, 5
            )
            atom = local_atom_threshold(
                transition_system, dfa, prior, 5
            )
            thresholds.append({
                "family": f"hub_m{m}",
                "seed": seed,
                "H": 5,
                **atom,
                "minimax_value": minimum_value,
                "minimax_root": minimum_root,
            })
            if m == 8:
                grid_rows = solve_grid(
                    transition_system, dfa, prior, 5, HUB_GRID
                )
                for row in grid_rows:
                    hub_plateau.append({
                        "seed": seed,
                        "alpha": row["alpha"],
                        "dynamic_value": row["dynamic_value"],
                        "minimax_value": row["minimax_value"],
                        "at_minimax_value": row["at_minimax_value"],
                    })
                if seed == 0:
                    for row in solve_grid(
                        transition_system, dfa, prior, 5, REPRESENTATIVE_GRID
                    ):
                        representative.append({
                            "family": "hub_m8_seed0",
                            **row,
                        })

    intel_system, _ = intel_lab_topological_pkwts()
    intel_dfa = firefighting_dfa()
    intel_prior = intel_two_mode_prior(intel_system)
    intel_minimax, intel_root = minimax(
        intel_system, intel_dfa, intel_prior, 11
    )
    thresholds.append({
        "family": "intel_scLTL",
        "seed": 0,
        "H": 11,
        **local_atom_threshold(intel_system, intel_dfa, intel_prior, 11),
        "minimax_value": intel_minimax,
        "minimax_root": intel_root,
    })
    for row in solve_grid(
        intel_system, intel_dfa, intel_prior, 11, REPRESENTATIVE_GRID
    ):
        representative.append({"family": "intel_scLTL", **row})

    write_rows(
        output_dir / "stage3_3_p1_hub_m8_plateau_10seeds.csv", hub_plateau
    )
    write_rows(
        output_dir / "stage3_3_p1_minimax_plateau_sweep.csv", representative
    )
    write_rows(
        output_dir / "stage3_3_p1_minimax_thresholds.csv", thresholds
    )
    print(
        "[PASS:RECOMPUTED] Stage 3.3 P1 endpoints: "
        f"{len(hub_plateau)} hub grid rows, {len(thresholds)} thresholds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
