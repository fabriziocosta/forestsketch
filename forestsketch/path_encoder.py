"""Decision-path feature extraction."""

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class DecisionPathEncoder(BaseEstimator, TransformerMixin):
    """Encode all visited nodes of a fitted scikit-learn forest."""

    def fit(self, forest, X=None, y=None):
        if not hasattr(forest, "decision_path") or not hasattr(
            forest, "estimators_"
        ):
            raise TypeError(
                "forest must be a fitted estimator with decision_path "
                "and estimators_"
            )
        self.forest_ = forest
        self.n_output_features_ = int(
            sum(estimator.tree_.node_count for estimator in forest.estimators_)
        )
        return self

    def transform(self, X):
        check_is_fitted(self, ("forest_", "n_output_features_"))
        X = check_array(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            ensure_2d=True,
        )
        decision_path = self.forest_.decision_path(X)
        indicator = decision_path[0] if isinstance(decision_path, tuple) else decision_path
        return sparse.csr_matrix(indicator, dtype=np.float64)
