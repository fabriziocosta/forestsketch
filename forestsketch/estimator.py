"""The public Forest Sketch transformer."""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.validation import check_array, check_is_fitted

from .normalizers import make_normalizer
from .path_encoder import DecisionPathEncoder
from .projectors import make_projector


class ForestSketchEstimator(BaseEstimator, TransformerMixin):
    """Iteratively encode random-forest paths into compact representations."""

    def __init__(
        self,
        estimator,
        n_components=64,
        n_iterations=1,
        initial_projection_type="sparse",
        path_projection_type="gaussian",
        concat_projection_type="sparse",
        normalization="none",
        path_encoder=None,
        normalizer=None,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.initial_projection_type = initial_projection_type
        self.path_projection_type = path_projection_type
        self.concat_projection_type = concat_projection_type
        self.normalization = normalization
        self.path_encoder = path_encoder
        self.normalizer = normalizer
        self.random_state = random_state

    def fit(self, X, y):
        X = self._validate_X(X)
        self._validate_parameters()
        self._validate_forest_estimator()
        if y is None:
            raise ValueError(
                "y is required because estimator must be a fitted random-forest "
                "classifier or regressor"
            )
        if X.shape[0] != len(y):
            raise ValueError("X and y have inconsistent numbers of samples")

        self.n_features_in_ = X.shape[1]
        self.input_normalizer_ = self._make_normalizer()
        X_for_projection = self.input_normalizer_.fit_transform(X)

        projection_seeds = self._projection_seeds(1 + 2 * self.n_iterations)
        self.initial_projector_ = make_projector(
            self.n_components,
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
            forest.fit(X_current, y)

            encoder = (
                clone(self.path_encoder)
                if self.path_encoder is not None
                else DecisionPathEncoder()
            )
            encoder.fit(forest)
            V = encoder.transform(X_current)

            path_normalizer = self._make_normalizer()
            V_normalized = path_normalizer.fit_transform(V)
            path_projector = make_projector(
                self.n_components,
                self.path_projection_type,
                projection_seeds[1 + 2 * iteration],
            )
            path_projector.fit(V_normalized)
            Z = path_projector.transform(V_normalized)

            C = _hstack(X_current, Z)
            concat_normalizer = self._make_normalizer()
            C_normalized = concat_normalizer.fit_transform(C)
            concat_projector = make_projector(
                self.n_components,
                self.concat_projection_type,
                projection_seeds[2 + 2 * iteration],
            )
            concat_projector.fit(C_normalized)
            X_current = concat_projector.transform(C_normalized)

            self.forests_.append(forest)
            self.path_encoders_.append(encoder)
            self.path_normalizers_.append(path_normalizer)
            self.path_projectors_.append(path_projector)
            self.concat_normalizers_.append(concat_normalizer)
            self.concat_projectors_.append(concat_projector)

        self.n_components_out_ = self.n_components
        self.projection_matrices_ = {
            "P0": getattr(self.initial_projector_, "components_", None),
            "PZ": [
                getattr(projector, "components_", None)
                for projector in self.path_projectors_
            ],
            "PC": [
                getattr(projector, "components_", None)
                for projector in self.concat_projectors_
            ],
        }
        return self

    def transform(self, X):
        X_current, _ = self._transform(X, return_intermediates=False)
        return X_current

    def transform_with_intermediates(self, X):
        """Return the final representation and a per-iteration diagnostic trace."""

        return self._transform(X, return_intermediates=True)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        if input_features is not None and len(input_features) != self.n_features_in_:
            raise ValueError(
                "input_features must have the same length as the fitted input"
            )
        return np.asarray(
            [f"forestsketch_{index}" for index in range(self.n_components_out_)],
            dtype=object,
        )

    def _transform(self, X, return_intermediates):
        check_is_fitted(
            self,
            (
                "initial_projector_",
                "forests_",
                "path_encoders_",
                "concat_projectors_",
            ),
        )
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X.shape[1]} features, but ForestSketchEstimator was fitted "
                f"with {self.n_features_in_} features"
            )

        X_for_projection = self.input_normalizer_.transform(X)
        X_current = self.initial_projector_.transform(X_for_projection)
        trace = []

        for iteration, encoder in enumerate(self.path_encoders_):
            V = encoder.transform(X_current)
            V_normalized = self.path_normalizers_[iteration].transform(V)
            Z = self.path_projectors_[iteration].transform(V_normalized)
            C = _hstack(X_current, Z)
            C_normalized = self.concat_normalizers_[iteration].transform(C)
            X_next = self.concat_projectors_[iteration].transform(C_normalized)

            if return_intermediates:
                trace.append(
                    {
                        "X": X_current,
                        "V": V,
                        "Z": Z,
                        "C": C,
                        "X_next": X_next,
                    }
                )
            X_current = X_next

        return X_current, trace

    def _make_normalizer(self):
        if self.normalizer is not None:
            return clone(self.normalizer)
        return make_normalizer(self.normalization)

    def _projection_seeds(self, count):
        rng = np.random.RandomState(self.random_state)
        return rng.randint(0, np.iinfo(np.int32).max, size=count)

    def _validate_parameters(self):
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        if not isinstance(self.n_iterations, (int, np.integer)) or self.n_iterations < 0:
            raise ValueError("n_iterations must be a non-negative integer")

    def _validate_forest_estimator(self):
        if not isinstance(
            self.estimator,
            (RandomForestClassifier, RandomForestRegressor),
        ):
            raise TypeError(
                "estimator must be a RandomForestClassifier or "
                "RandomForestRegressor"
            )

    @staticmethod
    def _validate_X(X):
        return check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )


def _hstack(left, right):
    if sparse.issparse(left) or sparse.issparse(right):
        return sparse.hstack((left, right), format="csr")
    return np.hstack((left, right))
