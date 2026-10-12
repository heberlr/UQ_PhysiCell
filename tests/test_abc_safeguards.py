#!/usr/bin/env python
"""Tests for the scaled ABC transition kernel and the between-generation safeguards."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pyabc import MultivariateNormalTransition

import uq_physicell.abc  # noqa: F401 -- applies pyABC patches
from uq_physicell.abc.safeguards import (
    ABCSafeguardError,
    check_population_weights,
    check_transition_support,
)
from uq_physicell.abc.transitions import ScaledMultivariateNormalTransition


def _mixed_scale_population(n=200, seed=0):
    """Four parameters spanning ~10 orders of magnitude, like ex11's ModelC."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        'apoptosis_rate': rng.uniform(4.25e-5, 6.38e-5, n),
        'cell_cycle_entry': rng.uniform(0, 5.76e-4, n),
        'oxygen_sat': rng.uniform(5.76e-4, 8.64e-4, n),
        'oxygen_hfm': rng.uniform(10, 30, n),
    })
    return X, np.ones(n) / n


class TestScaledMultivariateNormalTransition:
    def test_mixed_scales_keep_full_rank_and_positive_density(self):
        X, w = _mixed_scale_population()
        np.random.seed(1)
        plain = MultivariateNormalTransition()
        plain.fit(X, w)
        scaled = ScaledMultivariateNormalTransition()
        scaled.fit(X, w)
        # The failure being fixed: pyABC's kernel collapses and gives zero density
        assert np.mean(plain.pdf(plain.rvs(200)) == 0) > 0.5
        assert scaled.normal.cov_object.rank == X.shape[1]
        assert np.all(scaled.pdf(scaled.rvs(200)) > 0)

    def test_density_matches_pyabc_in_one_dimension(self):
        # Rescaling cannot change a 1D kernel's shape, so densities must be identical
        rng = np.random.default_rng(0)
        X = pd.DataFrame({'a': rng.normal(5e-5, 1e-5, 300)})
        w = np.ones(300) / 300
        plain = MultivariateNormalTransition()
        plain.fit(X, w)
        scaled = ScaledMultivariateNormalTransition()
        scaled.fit(X, w)
        xs = pd.DataFrame({'a': np.linspace(1e-5, 9e-5, 9)})
        np.testing.assert_allclose(scaled.pdf(xs), plain.pdf(xs), rtol=1e-10)

    def test_density_integrates_to_one_in_original_units(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame({'a': rng.normal(5e-5, 1e-5, 200), 'b': rng.normal(20, 5, 200)})
        scaled = ScaledMultivariateNormalTransition()
        scaled.fit(X, np.ones(200) / 200)
        ga, gb = np.linspace(-1e-5, 1.1e-4, 250), np.linspace(-15, 55, 250)
        A, B = np.meshgrid(ga, gb)
        density = scaled.pdf(pd.DataFrame({'a': A.ravel(), 'b': B.ravel()}))
        assert density.sum() * (ga[1] - ga[0]) * (gb[1] - gb[0]) == pytest.approx(1.0, abs=1e-3)

    def test_samples_are_in_original_units_and_columns(self):
        X, w = _mixed_scale_population()
        scaled = ScaledMultivariateNormalTransition()
        scaled.fit(X, w)
        df = scaled.rvs(500)
        assert list(df.columns) == list(X.columns)
        np.testing.assert_allclose(df.mean(), X.mean(), rtol=0.2)
        single = scaled.rvs_single()
        assert set(single.keys()) == set(X.columns)

    def test_single_particle_does_not_fail(self):
        X = pd.DataFrame({'a': [5e-5], 'b': [20.0]})
        scaled = ScaledMultivariateNormalTransition()
        scaled.fit(X, np.array([1.0]))
        assert np.isfinite(scaled.pdf(scaled.rvs(5))).all()


def _fake_abc_smc(weights_by_model, transitions=None, t=0):
    abc_smc = MagicMock()
    abc_smc.models = [MagicMock(name=f"m{m}") for m in range(3)]
    for m, model in enumerate(abc_smc.models):
        model.name = f"Model{m}"
    abc_smc.history.alive_models.return_value = sorted(weights_by_model)
    abc_smc.history.get_distribution.side_effect = (
        lambda m, t: (pd.DataFrame({'a': np.zeros(len(weights_by_model[m]))}), weights_by_model[m])
    )
    abc_smc.transitions = transitions or {}
    return abc_smc


class TestCheckPopulationWeights:
    def test_healthy_weights_pass(self):
        abc_smc = _fake_abc_smc({0: np.full(10, 0.1)})
        check_population_weights(abc_smc, 1, max_zero_weight_fraction=0.05)

    def test_all_weights_zeroed_raises_and_names_model(self):
        # ex11's ModelC: every weight zeroed, which normalizes to nan
        abc_smc = _fake_abc_smc({0: np.full(10, 0.1), 2: np.full(44, np.nan)})
        with pytest.raises(ABCSafeguardError, match="44 of 44 .*'Model2'"):
            check_population_weights(abc_smc, 1, max_zero_weight_fraction=0.05)

    def test_few_zeroed_weights_only_warn(self, caplog):
        w = np.full(100, 1 / 98)
        w[:2] = 0.0
        abc_smc = _fake_abc_smc({0: w})
        with caplog.at_level("WARNING"):
            check_population_weights(abc_smc, 1, max_zero_weight_fraction=0.05)
        assert "2 of 100" in caplog.text


class TestCheckTransitionSupport:
    def test_degenerate_kernel_raises_before_generation(self):
        X, w = _mixed_scale_population()
        plain = MultivariateNormalTransition()
        plain.fit(X, w)
        abc_smc = _fake_abc_smc({0: w}, transitions={0: plain})
        with pytest.raises(ABCSafeguardError, match="zero density"):
            check_transition_support(abc_smc, 1)

    def test_scaled_kernel_passes_and_keeps_random_state(self):
        X, w = _mixed_scale_population()
        scaled = ScaledMultivariateNormalTransition()
        scaled.fit(X, w)
        abc_smc = _fake_abc_smc({0: w}, transitions={0: scaled})
        np.random.seed(42)
        expected = np.random.rand()
        np.random.seed(42)
        check_transition_support(abc_smc, 1)
        assert np.random.rand() == expected
