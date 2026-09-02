
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from stage0_solver import AgentNode, EnvNode, Stage0RegretSolver, World
from stage1_common import normalize_prior
from stage1_3_risk import StaticCVaREvaluation, evaluate_policy_cvar


@dataclass
class CandidatePolicyCVaRResult:
    alpha: float
    selected_name: str
    selected_policy: Mapping[AgentNode, EnvNode]
    selected_evaluation: StaticCVaREvaluation
    evaluations: Dict[str, StaticCVaREvaluation]


class StaticCVaRCandidateSelector:
    """Exact selector over an explicitly supplied finite policy family.

    This is intentionally a Stage-1.3 *precommitment evaluator/ground truth*
    rather than the final scalable synthesis algorithm. Stage 1.4 will address
    full policy synthesis and time-consistent dynamic risk.

    The selector is exact over the supplied deterministic contingent policies:
        min_pi CVaR_alpha(J_pi(theta) - J*(theta)).
    """

    def __init__(
        self,
        solver: Stage0RegretSolver,
        prior: Mapping[World, float],
        alpha: float,
    ):
        self.solver = solver
        self.prior = normalize_prior(solver.T, prior)
        self.alpha = float(alpha)

    def solve(
        self,
        policies: Mapping[str, Mapping[AgentNode, EnvNode]],
    ) -> CandidatePolicyCVaRResult:
        if not policies:
            raise ValueError("At least one candidate policy is required.")

        evaluations: Dict[str, StaticCVaREvaluation] = {}
        ranked = []

        for name, policy in policies.items():
            ev = evaluate_policy_cvar(
                self.solver, policy, self.prior, self.alpha
            )
            evaluations[name] = ev
            ranked.append(
                (
                    ev.cvar_regret,
                    ev.expected_regret,
                    ev.worst_regret,
                    name,
                )
            )

        ranked.sort()
        selected_name = ranked[0][3]
        return CandidatePolicyCVaRResult(
            alpha=self.alpha,
            selected_name=selected_name,
            selected_policy=policies[selected_name],
            selected_evaluation=evaluations[selected_name],
            evaluations=evaluations,
        )
