"""Independent NumPy affine-Gaussian twin of the default CV channel.

The PhotoGraphiQ channel is interrogated once at construction to identify its
affine matrices.  Stepping thereafter uses only NumPy.  This is intentionally
limited to input-independent covariance resources; modulation is a different,
time-varying Gaussian channel and must be represented explicitly.
"""

from dataclasses import replace

import numpy as np

from .config import CVConfig
from .cv import CVMBReservoir


class GaussianClassicalTwin:
    """mu'=A mu+B u+c, V'=A V A^T+N for a fixed Gaussian CV reservoir."""

    def __init__(self, config: CVConfig | None = None):
        self.config = config or CVConfig()
        if self.config.phase_scale or self.config.squeeze_scale or self.config.adaptive_angle:
            raise NotImplementedError("Twin extraction currently requires a fixed Gaussian channel")
        self._oracle = CVMBReservoir(replace(self.config, readout_mode="state_oracle"))
        self._extract()
        self.reset()

    def _apply(self, mean, covariance, value):
        import photographiq as pg

        memory = pg.GaussianState(mean, covariance, self._oracle.nodes)
        return self._oracle.channel.apply(self._oracle._joint(memory, np.asarray([value], float)))

    def _extract(self):
        d = 2 * self.config.memory_modes
        base = self._apply(np.zeros(d), np.eye(d), 0.0)
        self.c = base.mean
        self.A = np.empty((d, d))
        for index in range(d):
            vector = np.zeros(d); vector[index] = 1.0
            self.A[:, index] = self._apply(vector, np.eye(d), 0.0).mean - self.c
        self.B = (self._apply(np.zeros(d), np.eye(d), 1.0).mean - self.c).reshape(d, 1)
        self.N = (base.covariance - self.A @ self.A.T + (base.covariance - self.A @ self.A.T).T) / 2

    def reset(self, *, mean=None, covariance=None):
        d = len(self.c)
        self.mean = np.zeros(d) if mean is None else np.asarray(mean, float).copy()
        self.covariance = np.eye(d) if covariance is None else np.asarray(covariance, float).copy()
        return self

    def step(self, value):
        encoded = self.config.input_scale * (self._oracle.mask @ np.asarray([value], float)) + self._oracle.bias
        # B is measured for a unit raw input; use channel identification rather
        # than PhotoGraphiQ after construction.
        self.mean = self.A @ self.mean + self.B[:, 0] * float(value) + self.c
        self.covariance = self.A @ self.covariance @ self.A.T + self.N
        return self.features()

    def features(self):
        preset = self.config.feature_preset
        diagonal = np.diag(self.covariance)
        if preset == "tier_a":
            return np.r_[self.mean, diagonal]
        if preset == "tier_b":
            return np.r_[self.mean, self.covariance[np.triu_indices(len(self.mean))]]
        if preset == "diagnostic_quadratic":
            return np.r_[self.mean, diagonal, self.mean**2 + diagonal]
        raise NotImplementedError("The redundant diagnostic preset is intentionally oracle-only")
