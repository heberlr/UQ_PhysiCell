import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from SALib.sample import sobol as sobol_sample

from uq_physicell.model_analysis.sensitivity_analysis import get_sobol_convergence


def _make_params_dict(samples_matrix, param_names):
    return {
        'samples': {i: dict(zip(param_names, row)) for i, row in enumerate(samples_matrix)},
        param_names[0]: {'lower_bound': 0.0, 'upper_bound': 1.0, 'ref_value': 0.5},
        param_names[1]: {'lower_bound': 0.0, 'upper_bound': 1.0, 'ref_value': 0.5},
    }


def _make_df_qois(samples_matrix, qoi_name, time_value, y):
    n = len(samples_matrix)
    return pd.DataFrame(
        {qoi_name: y},
        index=pd.MultiIndex.from_arrays(
            [range(n), [time_value] * n], names=['SampleID', 'time']
        ),
    )


@pytest.fixture
def sobol_design():
    problem = {'num_vars': 2, 'names': ['p1', 'p2'], 'bounds': [(0.0, 1.0), (0.0, 1.0)]}
    return sobol_sample.sample(problem, 32, calc_second_order=True, seed=42), ['p1', 'p2']


def test_get_sobol_convergence_isolates_the_influential_parameter(sobol_design):
    samples_matrix, param_names = sobol_design
    # Y depends only on p1 -- ST(p1) should be near 1, ST(p2) should be near 0,
    # at every N, confirming the prefix-truncation and per-N re-analysis wiring
    # is correct end to end.
    y = samples_matrix[:, 0]
    df_qois = _make_df_qois(samples_matrix, 'qoi', time_value=100.0, y=y)

    with patch(
        "uq_physicell.model_analysis.sensitivity_analysis.get_global_SA_parameters",
        return_value=_make_params_dict(samples_matrix, param_names),
    ):
        convergence = get_sobol_convergence('dummy.db', ['qoi'], df_qois, 100.0, [4, 8, 16, 32])

    assert list(convergence.keys()) == ['qoi']
    st_p1 = convergence['qoi']['p1']['ST']
    st_p2 = convergence['qoi']['p2']['ST']
    assert len(st_p1) == len(st_p2) == 4
    # Small N gives a noisy estimate -- that noise is exactly what this function
    # exists to surface -- so only require the influential parameter to clearly
    # dominate at every N, and require the estimate to have actually tightened
    # up by the largest N.
    for p1, p2 in zip(st_p1, st_p2):
        assert p1 > p2
    assert st_p1[-1] == pytest.approx(1.0, abs=0.15)
    assert st_p2[-1] == pytest.approx(0.0, abs=0.15)


def test_get_sobol_convergence_rejects_time_value_not_present(sobol_design):
    samples_matrix, param_names = sobol_design
    df_qois = _make_df_qois(samples_matrix, 'qoi', time_value=100.0, y=samples_matrix[:, 0])

    with patch(
        "uq_physicell.model_analysis.sensitivity_analysis.get_global_SA_parameters",
        return_value=_make_params_dict(samples_matrix, param_names),
    ):
        with pytest.raises(ValueError, match="not found"):
            get_sobol_convergence('dummy.db', ['qoi'], df_qois, 999.0, [4, 8])


def test_get_sobol_convergence_rejects_n_larger_than_available_samples(sobol_design):
    samples_matrix, param_names = sobol_design
    df_qois = _make_df_qois(samples_matrix, 'qoi', time_value=100.0, y=samples_matrix[:, 0])

    with patch(
        "uq_physicell.model_analysis.sensitivity_analysis.get_global_SA_parameters",
        return_value=_make_params_dict(samples_matrix, param_names),
    ):
        with pytest.raises(ValueError, match="needs"):
            get_sobol_convergence('dummy.db', ['qoi'], df_qois, 100.0, [4, 64])


def test_get_sobol_convergence_requires_time_index(sobol_design):
    samples_matrix, param_names = sobol_design
    df_qois = pd.DataFrame({'qoi': samples_matrix[:, 0]})  # no 'time' in the index

    with patch(
        "uq_physicell.model_analysis.sensitivity_analysis.get_global_SA_parameters",
        return_value=_make_params_dict(samples_matrix, param_names),
    ):
        with pytest.raises(ValueError, match="'SampleID', 'time'"):
            get_sobol_convergence('dummy.db', ['qoi'], df_qois, 100.0, [4])
