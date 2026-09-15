"""Sparse-safe normalization components."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_is_fitted


class IdentityNormalizer(BaseEstimator, TransformerMixin):
    """A no-op normalizer."""

    def fit(self, X, y=None):
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X.shape[1]} features, but this normalizer was fitted with "
                f"{self.n_features_in_} features"
            )
        return X


class RowNormalizer(BaseEstimator, TransformerMixin):
    """L1 or L2 row normalization that preserves sparse structure."""

    def __init__(self, norm="l2"):
        self.norm = norm

    def fit(self, X, y=None):
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        if self.norm not in {"l1", "l2"}:
            raise ValueError("norm must be 'l1' or 'l2'")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has "
                f"{X.shape[1]} features, but this normalizer was fitted with "
                f"{self.n_features_in_} features"
            )
        return normalize(X, norm=self.norm, axis=1, copy=True)


def make_normalizer(normalization):
    """Build a normalizer from a public normalization name."""

    if normalization == "none":
        return IdentityNormalizer()
    if normalization in {"l1", "l2"}:
        return RowNormalizer(norm=normalization)
    raise ValueError("normalization must be 'none', 'l1', or 'l2'")
