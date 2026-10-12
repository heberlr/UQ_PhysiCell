"""Transition kernels for ABC-SMC."""

import numpy as np
import pandas as pd
import scipy.stats as st
from pyabc import MultivariateNormalTransition
from pyabc.parameters import Parameter
from pyabc.transition.exceptions import NotEnoughParticles
from pyabc.transition.util import smart_cov


def _parameter_scales(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-parameter scale: the weighted standard deviation, falling back to the
    weighted mean's magnitude (a single or collapsed particle) and then to 1."""
    mean = np.average(X, axis=0, weights=w)
    std = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=w))
    scale = np.where(np.isfinite(std) & (std > 0), std, np.abs(mean))
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


class ScaledMultivariateNormalTransition(MultivariateNormalTransition):
    """Multivariate normal kernel fitted on standardized parameters.

    pyABC's ``MultivariateNormalTransition`` fits its covariance in the parameters'
    own units. When parameters differ by orders of magnitude (e.g. a rate of 1e-5
    1/min next to a half-max of 20 mmHg), the covariance spans so many orders of
    magnitude that ``scipy.stats.multivariate_normal(..., allow_singular=True)``
    treats the small directions as zero variance. Its density is then exactly 0 for
    almost every new proposal, every importance weight becomes ``prior / 0 = inf``,
    and the model is silently lost.

    This kernel divides each parameter by its weighted standard deviation before
    fitting, samples in that standardized space, and maps samples back. The density
    is returned in the original units (divided by the product of the scales), so
    importance weights and model probabilities are unchanged in meaning: only the
    shape of the proposal differs, and it now perturbs every parameter in
    proportion to its own spread.
    """

    def fit(self, X: pd.DataFrame, w: np.ndarray) -> None:
        if len(X) == 0:
            raise NotEnoughParticles('Fitting not possible.')
        self.scale_ = _parameter_scales(X.values, w)
        self._jacobian = float(np.prod(self.scale_))
        self._X_arr = X.values / self.scale_

        sample_cov = smart_cov(self._X_arr, w)
        dim = sample_cov.shape[0]
        eff_sample_size = 1 / (w**2).sum()
        bw_factor = self.bandwidth_selector(eff_sample_size, dim)

        self.cov = sample_cov * bw_factor**2 * self.scaling
        self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

        # cache range array
        self._range = np.arange(len(self.X))

    def rvs(self, size: int = None):
        if size is None:
            return self.rvs_single()
        sample_ind = np.random.choice(self._range, size=size, p=self.w, replace=True)
        perturbed = self._X_arr[sample_ind] + np.random.multivariate_normal(
            np.zeros(self.cov.shape[0]), self.cov, size=size
        )
        return pd.DataFrame(perturbed * self.scale_, columns=self.X.columns)

    def rvs_single(self) -> Parameter:
        sample_ind = np.random.choice(self._range, p=self.w, replace=True)
        perturbed = self._X_arr[sample_ind] + np.random.multivariate_normal(
            np.zeros(self.cov.shape[0]), self.cov
        )
        return Parameter(dict(zip(self.X.columns, perturbed * self.scale_)))

    def _pdf_single(self, x: np.ndarray) -> float:
        # Density in standardized space, converted back to the original units
        density = (self.normal.pdf(x / self.scale_ - self._X_arr) * self.w).sum()
        return float(density / self._jacobian)
