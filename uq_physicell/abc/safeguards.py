"""Run-time checks that stop an ABC-SMC run early instead of letting it silently
produce wrong model probabilities or crash generations later."""

import logging

import numpy as np
from pyabc import ABCSMC

logger = logging.getLogger(__name__)


class ABCSafeguardError(RuntimeError):
    """An ABC-SMC run was stopped because continuing would give wrong results."""


def _model_name(abc_smc: ABCSMC, m: int) -> str:
    name = getattr(abc_smc.models[m], 'name', None)
    return f"'{name}' (model {m})" if name else f"model {m}"


def check_population_weights(abc_smc: ABCSMC, t: int, max_zero_weight_fraction: float) -> None:
    """Check the importance weights each alive model got in generation ``t``.

    A weight of exactly zero means the particle's weight was non-finite and was
    replaced by ``patch_pyabc_nan_particle_weight`` (a genuine importance weight is
    always positive). A few such particles are tolerated with a warning; above
    ``max_zero_weight_fraction`` of a model's particles the run is stopped, because
    that model's probability and posterior would be wrong.

    Raises:
        ABCSafeguardError: If a model has too many zeroed (or non-finite) weights.
    """
    for m in abc_smc.history.alive_models(t):
        _, w = abc_smc.history.get_distribution(m, t)
        n = len(w)
        # all-zero weights normalize to nan, so count both
        n_bad = int(np.sum(~np.isfinite(w) | (w == 0)))
        if n == 0 or n_bad == 0:
            continue
        message = (
            f"{n_bad} of {n} accepted particles of {_model_name(abc_smc, m)} at generation "
            f"t={t} have a zero or non-finite importance weight: the transition density "
            f"was zero where they were proposed."
        )
        if n_bad / n > max_zero_weight_fraction:
            raise ABCSafeguardError(
                f"ABC-SMC stopped. {message} Their contribution to the model probabilities "
                f"and posterior would be wrong. Usual cause: a transition kernel that cannot "
                f"represent parameters on very different scales (use the default "
                f"transition_strategy='multivariate'). Generations up to t={t} are stored in "
                f"the database, but t={t} itself is affected."
            )
        logger.warning(f"{message} They were given zero weight.")


def check_transition_support(abc_smc: ABCSMC, t: int, n_draws: int = 200,
                             max_zero_density_fraction: float = 0.05) -> None:
    """Check that each fitted transition kernel gives positive density to its own proposals.

    A kernel whose density is zero where it proposes would give every particle of that
    model an infinite (then zeroed) weight. Checked right after fitting, before any
    simulation of generation ``t`` runs. The global NumPy random state is restored so
    the check does not change the run's random stream.

    Raises:
        ABCSafeguardError: If a kernel gives zero or non-finite density to too many of its own proposals.
    """
    state = np.random.get_state()
    try:
        for m in abc_smc.history.alive_models(t - 1):
            transition = abc_smc.transitions[m]
            if transition.no_meaningful_particles():
                continue
            density = np.atleast_1d(transition.pdf(transition.rvs(size=n_draws)))
            n_bad = int(np.sum(~np.isfinite(density) | (density <= 0)))
            if n_bad / len(density) > max_zero_density_fraction:
                raise ABCSafeguardError(
                    f"ABC-SMC stopped before generation t={t}: the transition kernel of "
                    f"{_model_name(abc_smc, m)} gives zero density to {n_bad} of {len(density)} "
                    f"of its own proposals, so their importance weights would be infinite. "
                    f"Usual cause: parameters on very different scales "
                    f"({type(transition).__name__}). Use the default "
                    f"transition_strategy='multivariate', which standardizes parameters."
                )
    finally:
        np.random.set_state(state)


class CheckedABCSMC(ABCSMC):
    """``pyabc.ABCSMC`` that checks weights and transition kernels between generations.

    Before fitting the kernels for generation ``t`` it checks the weights of generation
    ``t-1`` (:func:`check_population_weights`); after fitting, it checks the kernels
    themselves (:func:`check_transition_support`). Both run in the main process and
    cost no simulations.

    Args:
        max_zero_weight_fraction (float): Largest share of a model's particles allowed
            to have a zeroed weight in one generation before the run stops. Default 0.05.
        *args, **kwargs: Passed to ``pyabc.ABCSMC``.
    """

    def __init__(self, *args, max_zero_weight_fraction: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_zero_weight_fraction = max_zero_weight_fraction

    def _fit_transitions(self, t):
        if t > 0:
            check_population_weights(self, t - 1, self.max_zero_weight_fraction)
        super()._fit_transitions(t)
        if t > 0:
            check_transition_support(self, t)
