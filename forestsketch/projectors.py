"""Projection components used by Forest Sketch."""

from numbers import Integral

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted


class SklearnRandomProjector(BaseEstimator, TransformerMixin):
    """Adapter around scikit-learn's materialized random projections."""

    def __init__(self, n_components, projection_type="gaussian", random_state=None):
        self.n_components = n_components
        self.projection_type = projection_type
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        self._validate_n_components()
        if self.projection_type not in {"gaussian", "sparse"}:
            raise ValueError(
                "projection_type must be 'gaussian' or 'sparse' "
                "for SklearnRandomProjector"
            )

        transformer_type = (
            GaussianRandomProjection
            if self.projection_type == "gaussian"
            else SparseRandomProjection
        )
        self._transformer = transformer_type(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self._transformer.fit(X)
        self.n_features_in_ = X.shape[1]
        self.components_ = self._transformer.components_
        return self

    def transform(self, X):
        check_is_fitted(self, ("_transformer", "components_", "n_features_in_"))
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        self._check_input_width(X)
        return self._transformer.transform(X)

    def _validate_n_components(self):
        if not isinstance(self.n_components, Integral) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer")

    def _check_input_width(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X.shape[1]} features, but this projector was fitted with "
                f"{self.n_features_in_} features"
            )


class SignedHashProjector(BaseEstimator, TransformerMixin):
    """Deterministic one-bucket signed hashing projection.

    Each input feature is assigned to one output bucket and a sign using a
    stable 64-bit mixing function. The projection is never materialized.
    """

    def __init__(self, n_components, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        if not isinstance(self.n_components, Integral) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        self._seed = int(rng.randint(0, np.iinfo(np.uint32).max))
        return self

    def transform(self, X):
        check_is_fitted(self, ("n_features_in_", "_seed"))
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X.shape[1]} features, but this projector was fitted with "
                f"{self.n_features_in_} features"
            )

        values = sparse.coo_matrix(X)
        mixed = self._mix_feature_ids(values.col)
        buckets = mixed % np.uint64(self.n_components)
        signs = np.where(
            (mixed & np.uint64(1)) == np.uint64(0),
            1.0,
            -1.0,
        )
        output = sparse.coo_matrix(
            (values.data * signs, (values.row, buckets)),
            shape=(X.shape[0], self.n_components),
        ).tocsr()
        output.sum_duplicates()
        return output

    def _mix_feature_ids(self, feature_ids):
        value = np.asarray(feature_ids, dtype=np.uint64)
        value = value + np.uint64(self._seed)
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
        return value


def make_projector(n_components, projection_type, random_state=None):
    """Build a projector from a public projection type name."""

    if projection_type == "signed_hash":
        return SignedHashProjector(
            n_components=n_components,
            random_state=random_state,
        )
    if projection_type in {"gaussian", "sparse"}:
        return SklearnRandomProjector(
            n_components=n_components,
            projection_type=projection_type,
            random_state=random_state,
        )
    raise ValueError(
        "projection_type must be 'gaussian', 'sparse', or 'signed_hash'"
    )
