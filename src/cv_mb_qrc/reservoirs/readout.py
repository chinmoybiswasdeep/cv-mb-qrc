"""Train-only standardization and an unpenalized ridge intercept."""

import numpy as np


class RidgeReadout:
    def __init__(
        self, regularization=1e-4, *, variance_floor=1e-12, input_access="reservoir_plus_input"
    ):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("Ridge regularization must be finite and positive")
        self.regularization = regularization
        if not np.isfinite(variance_floor) or variance_floor < 0:
            raise ValueError("variance_floor must be finite and nonnegative")
        self.variance_floor = variance_floor
        if input_access not in ("reservoir_only", "reservoir_plus_input"):
            raise ValueError("input_access must be reservoir_only or reservoir_plus_input")
        self.input_access = input_access
        self.weights = None

    def design(self, inputs, features):
        u, x = np.asarray(inputs, float), np.asarray(features, float)
        if u.ndim == 1:
            u = u[:, None]
        if u.ndim != 2 or x.ndim != 2 or len(u) != len(x):
            raise ValueError("Inputs/features must have matching sample axes")
        result = np.column_stack([u, x]) if self.input_access == "reservoir_plus_input" else x
        if not len(result) or not np.isfinite(result).all():
            raise ValueError("Design must be nonempty and finite")
        return result

    def fit(self, inputs, features, targets):
        x = self.design(inputs, features)
        y = np.asarray(targets, float)
        if y.ndim not in (1, 2) or len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError("Targets must be finite with matching sample axis")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        # Selection is fit strictly on training features, never validation/test.
        self.kept_columns = self.scale >= self.variance_floor
        self.removed_columns = np.flatnonzero(~self.kept_columns).tolist()
        if not np.any(self.kept_columns):
            raise ValueError("All readout columns collapsed on the training partition")
        self.mean, self.scale = self.mean[self.kept_columns], self.scale[self.kept_columns]
        self.scale[self.scale == 0] = 1
        z = (x[:, self.kept_columns] - self.mean) / self.scale
        self.target_mean = y.mean(axis=0)
        # SVD avoids squaring condition numbers and handles constant columns.
        u, s, vh = np.linalg.svd(z, full_matrices=False)
        multiplier = s / (s * s + self.regularization)
        self.weights = (vh.T * multiplier) @ (u.T @ (y - self.target_mean))
        return self

    def predict(self, inputs, features):
        if self.weights is None:
            raise ValueError("Fit readout on training data first")
        x = self.design(inputs, features)
        if x.shape[1] != len(self.kept_columns):
            raise ValueError("Feature dimension differs from training")
        return (x[:, self.kept_columns] - self.mean) / self.scale @ self.weights + self.target_mean
