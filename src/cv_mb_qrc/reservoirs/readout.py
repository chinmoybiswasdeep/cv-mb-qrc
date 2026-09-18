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


class MultiTargetRidgeReadout(RidgeReadout):
    """Per-target validation selection sharing one training design SVD."""

    def __init__(self, regularizations, **kwargs):
        regularizations = np.asarray(tuple(regularizations), float)
        if (
            regularizations.ndim != 1
            or not len(regularizations)
            or not np.isfinite(regularizations).all()
            or np.any(regularizations <= 0)
        ):
            raise ValueError("Regularizations must be finite and positive")
        super().__init__(float(regularizations[0]), **kwargs)
        self.regularizations = regularizations
        self.factorization_count = 0

    def fit_with_validation(self, train, validation):
        train_inputs, train_features, train_targets = train
        validation_inputs, validation_features, validation_targets = validation
        x = self.design(train_inputs, train_features)
        validation_x = self.design(validation_inputs, validation_features)
        y = np.asarray(train_targets, float)
        validation_y = np.asarray(validation_targets, float)
        if y.ndim == 1:
            y = y[:, None]
        if validation_y.ndim == 1:
            validation_y = validation_y[:, None]
        if len(y) != len(x) or validation_y.shape[1] != y.shape[1]:
            raise ValueError("Multi-target train/validation targets are not aligned")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.kept_columns = self.scale >= self.variance_floor
        self.removed_columns = np.flatnonzero(~self.kept_columns).tolist()
        if not np.any(self.kept_columns):
            raise ValueError("All readout columns collapsed on the training partition")
        self.mean, self.scale = self.mean[self.kept_columns], self.scale[self.kept_columns]
        self.scale[self.scale == 0] = 1
        z = (x[:, self.kept_columns] - self.mean) / self.scale
        validation_z = (validation_x[:, self.kept_columns] - self.mean) / self.scale
        self.target_mean = y.mean(axis=0)
        centered_targets = y - self.target_mean
        u, singular_values, vh = np.linalg.svd(z, full_matrices=False)
        self.factorization_count += 1
        projected_targets = u.T @ centered_targets
        candidates, losses = [], []
        for alpha in self.regularizations:
            multiplier = singular_values / (singular_values**2 + alpha)
            weights = (vh.T * multiplier) @ projected_targets
            candidates.append(weights)
            prediction = validation_z @ weights + self.target_mean
            losses.append(np.mean((prediction - validation_y) ** 2, axis=0))
        loss_matrix = np.asarray(losses)
        selected = np.argmin(loss_matrix, axis=0)
        self.selected_regularizations = self.regularizations[selected]
        self.weights = np.column_stack(
            [candidates[index][:, target] for target, index in enumerate(selected)]
        )
        return self
