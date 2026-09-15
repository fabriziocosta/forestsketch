"""The public Recursive Sketch transformer."""

from numbers import Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_X_y,
)

from .normalizers import make_normalizer
from .path_encoder import DecisionPathEncoder
from .projectors import make_projector


class RecursiveSketchClassifier(BaseEstimator, TransformerMixin):
    """Iteratively encode forest paths or transformer outputs into representations."""

    def __init__(
        self,
        estimator,
        n_components=64,
        n_iterations=1,
        dimension_mode="expanding",
        output_format="auto",
        initial_projection_type="sparse",
        path_projection_type="gaussian",
        concat_projection_type="sparse",
        normalization="none",
        path_encoder=None,
        normalizer=None,
        random_state=None,
        input_normalizer=None,
        path_normalizer=None,
        concat_normalizer=None,
        dimension_ratio=None,
    ):
        self.estimator = estimator
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.dimension_mode = dimension_mode
        self.output_format = output_format
        self.initial_projection_type = initial_projection_type
        self.path_projection_type = path_projection_type
        self.concat_projection_type = concat_projection_type
        self.normalization = normalization
        self.path_encoder = path_encoder
        self.normalizer = normalizer
        self.random_state = random_state
        self.input_normalizer = input_normalizer
        self.path_normalizer = path_normalizer
        self.concat_normalizer = concat_normalizer
        self.dimension_ratio = dimension_ratio

    def fit(self, X, y=None, sample_weight=None):
        self._fit(X, y, sample_weight=sample_weight, return_training_output=False)
        return self

    def fit_transform(self, X, y=None, sample_weight=None, **fit_params):
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {unexpected}")
        return self._fit(X, y, sample_weight=sample_weight, return_training_output=True)

    def _fit(self, X, y, sample_weight=None, return_training_output=False):
        self._validate_parameters()
        self._validate_path_estimator()
        if y is None:
            raise ValueError(
                "y is required because the path estimator is supervised"
            )

        feature_names = self._get_feature_names(X)
        X, y = check_X_y(
            X,
            y,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        if sample_weight is not None:
            check_consistent_length(X, y, sample_weight)
        else:
            check_consistent_length(X, y)
        self._set_feature_names(feature_names)

        self.n_features_in_ = X.shape[1]
        self.n_components_ = self._resolve_n_components(self.n_features_in_)
        self.input_normalizer_ = self._make_normalizer("input")
        X_for_projection = self._fit_transform_normalizer(self.input_normalizer_, X)

        projection_seeds = self._projection_seeds(1 + 2 * self.n_iterations)
        self.initial_projector_ = make_projector(
            self.n_components_,
            self.initial_projection_type,
            projection_seeds[0],
        )
        self.initial_projector_.fit(X_for_projection)
        X_current = self.initial_projector_.transform(X_for_projection)

        self.forests_ = []
        self.path_encoders_ = []
        self.path_normalizers_ = []
        self.path_projectors_ = []
        self.concat_normalizers_ = []
        self.concat_projectors_ = []

        for iteration in range(self.n_iterations):
            forest = clone(self.estimator)
            if sample_weight is None:
                forest.fit(X_current, y)
            else:
                forest.fit(X_current, y, sample_weight=sample_weight)

            if self.path_encoder is None and self._uses_estimator_transform(forest):
                # A generic path estimator owns both fitting and path-matrix
                # construction, e.g. RecursivePartitionClassifier.
                encoder = forest
                V = forest.transform(X_current)
            else:
                encoder = (
                    clone(self.path_encoder)
                    if self.path_encoder is not None
                    else DecisionPathEncoder()
                )
                encoder.fit(forest)
                V = encoder.transform(X_current)

            path_normalizer = self._make_normalizer("path")
            V_normalized = self._fit_transform_normalizer(path_normalizer, V)
            path_projector = make_projector(
                self.n_components_,
                self.path_projection_type,
                projection_seeds[1 + 2 * iteration],
            )
            path_projector.fit(V_normalized)
            Z = path_projector.transform(V_normalized)

            C = _hstack(X_current, Z)
            if self.dimension_mode == "fixed":
                concat_normalizer = self._make_normalizer("concat")
                C_normalized = self._fit_transform_normalizer(concat_normalizer, C)
                concat_projector = make_projector(
                    self.n_components_,
                    self.concat_projection_type,
                    projection_seeds[2 + 2 * iteration],
                )
                concat_projector.fit(C_normalized)
                X_current = concat_projector.transform(C_normalized)
            else:
                concat_normalizer = self._make_normalizer("concat")
                X_current = self._fit_transform_normalizer(concat_normalizer, C)
                concat_projector = None

            self.forests_.append(forest)
            self.path_encoders_.append(encoder)
            self.path_normalizers_.append(path_normalizer)
            self.path_projectors_.append(path_projector)
            self.concat_normalizers_.append(concat_normalizer)
            self.concat_projectors_.append(concat_projector)

        self.n_components_out_ = (
            self.n_components_
            if self.dimension_mode == "fixed"
            else self.n_components_ * (self.n_iterations + 1)
        )
        self.output_dimension_ = self.n_components_out_
        self.projection_matrices_ = {
            "P0": getattr(self.initial_projector_, "components_", None),
            "PZ": [
                getattr(projector, "components_", None)
                for projector in self.path_projectors_
            ],
            "PC": [
                getattr(projector, "components_", None)
                if projector is not None
                else None
                for projector in self.concat_projectors_
            ],
        }
        if return_training_output:
            return self._format_output(X_current)
        return None

    def transform(self, X):
        return self._transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            input_features = np.asarray(input_features, dtype=object)
            if input_features.ndim != 1:
                raise ValueError("input_features must be a 1-dimensional array")
            if input_features.shape[0] != self.n_features_in_:
                raise ValueError(
                    "input_features must have the same length as the fitted input"
                )
            if hasattr(self, "feature_names_in_") and not np.array_equal(
                input_features, self.feature_names_in_
            ):
                raise ValueError(
                    "input_features is not equal to the feature names seen during fit"
                )
        return np.asarray(
            [f"recursivesketch_{index}" for index in range(self.n_components_out_)],
            dtype=object,
        )

    def _transform(self, X):
        check_is_fitted(
            self,
            (
                "initial_projector_",
                "forests_",
                "path_encoders_",
                "concat_projectors_",
            ),
        )
        feature_names = self._get_feature_names(X)
        X = self._validate_X(X)
        self._check_feature_names(feature_names)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X.shape[1]} features, but RecursiveSketchClassifier was fitted "
                f"with {self.n_features_in_} features"
            )

        X_for_projection = self.input_normalizer_.transform(X)
        X_current = self.initial_projector_.transform(X_for_projection)

        for iteration, encoder in enumerate(self.path_encoders_):
            V = encoder.transform(X_current)
            V_normalized = self.path_normalizers_[iteration].transform(V)
            Z = self.path_projectors_[iteration].transform(V_normalized)
            C = _hstack(X_current, Z)
            if self.dimension_mode == "fixed":
                C_normalized = self.concat_normalizers_[iteration].transform(C)
                X_next = self.concat_projectors_[iteration].transform(C_normalized)
            else:
                X_next = self.concat_normalizers_[iteration].transform(C)

            X_current = X_next

        return self._format_output(X_current)

    def _make_normalizer(self, stage):
        stage_normalizer = getattr(self, f"{stage}_normalizer")
        template = stage_normalizer if stage_normalizer is not None else self.normalizer
        if template is None:
            return make_normalizer(self.normalization)
        try:
            normalizer = clone(template)
        except Exception as exc:
            raise TypeError(
                f"{stage}_normalizer must be a cloneable transformer with fit "
                "and transform methods"
            ) from exc
        if not hasattr(normalizer, "fit") or not hasattr(normalizer, "transform"):
            raise TypeError(
                f"{stage}_normalizer must provide fit and transform methods"
            )
        return normalizer

    def _projection_seeds(self, count):
        rng = np.random.RandomState(self.random_state)
        return rng.randint(0, np.iinfo(np.int32).max, size=count)

    @staticmethod
    def _fit_transform_normalizer(normalizer, X):
        if hasattr(normalizer, "fit_transform"):
            return normalizer.fit_transform(X)
        return normalizer.fit(X).transform(X)

    def _validate_parameters(self):
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if self.dimension_ratio is not None and (
            isinstance(self.dimension_ratio, (bool, np.bool_))
            or not isinstance(self.dimension_ratio, Real)
            or not np.isfinite(self.dimension_ratio)
            or self.dimension_ratio <= 0
        ):
            raise ValueError("dimension_ratio must be a positive finite number or None")
        if not isinstance(self.n_iterations, (int, np.integer)) or self.n_iterations < 0:
            raise ValueError("n_iterations must be a non-negative integer")
        if self.dimension_mode not in {"fixed", "expanding"}:
            raise ValueError("dimension_mode must be 'fixed' or 'expanding'")
        if self.output_format not in {"auto", "dense", "sparse"}:
            raise ValueError("output_format must be 'auto', 'dense', or 'sparse'")

    def _resolve_n_components(self, n_features):
        """Resolve the fitted projection width from the input feature width."""

        if self.dimension_ratio is None:
            return int(self.n_components)
        return max(1, int(np.ceil(self.dimension_ratio * n_features)))

    def _validate_path_estimator(self):
        if isinstance(self.estimator, (RandomForestClassifier, RandomForestRegressor)):
            return
        if self._uses_estimator_transform(self.estimator):
            return
        raise TypeError(
            "estimator must be a RandomForestClassifier, RandomForestRegressor, "
            "or an estimator implementing fit and transform"
        )

    @staticmethod
    def _uses_estimator_transform(estimator):
        return callable(getattr(estimator, "fit", None)) and callable(
            getattr(estimator, "transform", None)
        )

    @staticmethod
    def _validate_X(X):
        return check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )

    @staticmethod
    def _get_feature_names(X):
        """Return string column names from tabular input when available."""

        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        names = np.asarray(columns, dtype=object)
        if names.ndim != 1 or not all(isinstance(name, str) for name in names):
            return None
        return names

    def _set_feature_names(self, feature_names):
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

    def _check_feature_names(self, feature_names):
        if not hasattr(self, "feature_names_in_") or feature_names is None:
            return
        if not np.array_equal(feature_names, self.feature_names_in_):
            raise ValueError(
                "The feature names in the input do not match those seen during fit"
            )

    def _format_output(self, X):
        if self.output_format == "dense" and sparse.issparse(X):
            return X.toarray()
        if self.output_format == "sparse" and not sparse.issparse(X):
            return sparse.csr_matrix(X)
        return X


# Backward-compatible name retained for callers of releases before the
# Recursive Sketch rename.
ForestSketchEstimator = RecursiveSketchClassifier


def _hstack(left, right):
    if sparse.issparse(left) or sparse.issparse(right):
        return sparse.hstack((left, right), format="csr")
    return np.hstack((left, right))
