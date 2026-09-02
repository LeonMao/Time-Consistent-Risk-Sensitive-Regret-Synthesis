
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Dict, FrozenSet, Mapping, Sequence, Tuple

from stage0_solver import Knowledge, PKWTS, World


@dataclass(frozen=True)
class ProductComponent:
    weight: float
    # state -> categorical probability vector over T.patterns[state]
    marginals: Mapping[str, Tuple[float, ...]]


class MixtureProductPrior:
    """
    Exact factored prior:
        b(theta) = sum_c lambda_c prod_x p_c,x(theta_x)

    Unknown topology variables are the PK-WTS states with >1 successor pattern.
    Known states need no factor.

    C=1 gives an independent categorical prior.
    C>1 induces correlations through the latent mixture component.
    """

    def __init__(self, T: PKWTS, components: Sequence[ProductComponent]):
        if not components:
            raise ValueError("At least one product component is required.")
        self.T = T
        self.variables = tuple(x for x in T.states if len(T.patterns[x]) > 1)

        raw_weights = [float(c.weight) for c in components]
        if any(w < 0 for w in raw_weights) or sum(raw_weights) <= 0:
            raise ValueError("Component weights must be nonnegative and have positive sum.")
        z = sum(raw_weights)

        normalized = []
        for comp, w in zip(components, raw_weights):
            marg = {}
            for x in self.variables:
                if x not in comp.marginals:
                    raise ValueError(f"Missing marginal for variable state {x}.")
                p = tuple(float(v) for v in comp.marginals[x])
                if len(p) != len(T.patterns[x]):
                    raise ValueError(f"Wrong marginal size for {x}.")
                if any(v < 0 for v in p) or sum(p) <= 0:
                    raise ValueError(f"Invalid categorical marginal at {x}.")
                s = sum(p)
                marg[x] = tuple(v / s for v in p)
            normalized.append(ProductComponent(weight=w / z, marginals=marg))

        self.components = tuple(normalized)
        self._evidence_mass_cache: Dict[Knowledge, Tuple[float, ...]] = {}
        self._mode_cache: Dict[Knowledge, Tuple[float, ...]] = {}
        self._marginal_cache: Dict[Tuple[Knowledge, str], Tuple[float, ...]] = {}

    def component_evidence_masses(self, K: Knowledge) -> Tuple[float, ...]:
        """Return the unnormalized masses lambda_c L_c(K)."""
        if K in self._evidence_mass_cache:
            return self._evidence_mass_cache[K]

        kd = dict(K)
        masses = []
        for comp in self.components:
            m = comp.weight
            for x, pidx in kd.items():
                if x in comp.marginals:
                    m *= comp.marginals[x][pidx]
                    if m == 0.0:
                        break
            masses.append(m)

        out = tuple(masses)
        self._evidence_mass_cache[K] = out
        return out

    def component_posterior_weights(self, K: Knowledge) -> Tuple[float, ...]:
        if K in self._mode_cache:
            return self._mode_cache[K]

        masses = self.component_evidence_masses(K)
        z = sum(masses)
        if z <= 0:
            raise ValueError("Knowledge has zero probability under the factored prior.")

        post = tuple(m / z for m in masses)
        self._mode_cache[K] = post
        return post

    def marginal(self, K: Knowledge, x: str) -> Tuple[float, ...]:
        """Posterior P(theta_x = k | K)."""
        key = (K, x)
        if key in self._marginal_cache:
            return self._marginal_cache[key]

        if len(self.T.patterns[x]) == 1:
            out = (1.0,)
            self._marginal_cache[key] = out
            return out

        kd = dict(K)
        if x in kd:
            out = tuple(1.0 if k == kd[x] else 0.0 for k in range(len(self.T.patterns[x])))
            self._marginal_cache[key] = out
            return out

        mode_w = self.component_posterior_weights(K)
        out = []
        for k in range(len(self.T.patterns[x])):
            out.append(sum(
                mode_w[c] * self.components[c].marginals[x][k]
                for c in range(len(self.components))
            ))
        s = sum(out)
        out_t = tuple(v / s for v in out)
        self._marginal_cache[key] = out_t
        return out_t

    def possible_pattern_indices(self, K: Knowledge, x: str) -> Tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.marginal(K, x)) if p > 0.0)

    def observation_distribution(self, K: Knowledge, x: str) -> Dict[int, float]:
        return {
            i: p
            for i, p in enumerate(self.marginal(K, x))
            if p > 0.0
        }

    def condition(self, K: Knowledge, x: str, pidx: int) -> Knowledge:
        if len(self.T.patterns[x]) == 1:
            return K
        kd = dict(K)
        if x in kd and kd[x] != pidx:
            raise ValueError("Inconsistent observation.")
        K2 = frozenset(set(K) | {(x, pidx)})
        # Validate nonzero posterior mass.
        self.component_posterior_weights(K2)
        return K2

    def explicit_world_probability(self, world: World) -> float:
        """For validation only; does not enumerate worlds."""
        total = 0.0
        idx = self.T.state_index
        for comp in self.components:
            p = comp.weight
            for x in self.variables:
                p *= comp.marginals[x][world[idx[x]]]
            total += p
        return total

    def conceptual_world_upper_bound(self) -> int:
        """Product of pattern counts that have positive probability in at least one mode."""
        counts = []
        for x in self.variables:
            supported = set()
            for comp in self.components:
                for k, p in enumerate(comp.marginals[x]):
                    if p > 0:
                        supported.add(k)
            counts.append(len(supported))
        return prod(counts) if counts else 1

    def explicit_prior_for_validation(self) -> Dict[World, float]:
        """Enumerate the PK-WTS world product; use only for small validation cases."""
        out = {}
        for w in self.T.all_worlds():
            p = self.explicit_world_probability(w)
            if p > 0:
                out[w] = p
        z = sum(out.values())
        return {w: p / z for w, p in out.items()}
