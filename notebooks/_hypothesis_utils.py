"""Small, reproducible helpers shared by the hypothesis notebooks."""

from __future__ import annotations

import pickle
import time

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from forestsketch import DecisionPathEncoder, ForestSketchEstimator, make_projector


CLASSIFICATION_SEEDS = (0, 1, 2)


def classification_data(seed=42, n_samples=2200, n_features=120):
    """Return a fixed train/validation/test split for a difficult classification task."""
    n_informative = max(2, min(n_features - 1, 12))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=min(24, n_features - n_informative),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=0.65,
        flip_y=0.04,
        weights=[0.55, 0.45],
        random_state=seed,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=seed + 1000,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_data(seed=42, n_samples=1800, n_features=80):
    """Return a fixed train/validation/test split for a noisy regression task."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=16,
        n_targets=1,
        noise=18.0,
        random_state=seed,
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=seed + 1000
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def forest_classifier(seed, n_estimators=40):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        n_jobs=-1,
        random_state=seed,
    )


def forest_regressor(seed, n_estimators=40):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        n_jobs=-1,
        random_state=seed,
    )


def downstream_classifier(seed=42):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=seed),
    )


def downstream_regressor():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def initial_projection(X_train, X_test, n_components, seed, projection_type="sparse"):
    projector = make_projector(n_components, projection_type, seed)
    projector.fit(X_train)
    return projector.transform(X_train), projector.transform(X_test), projector


class OneShotTreePathTransformer(BaseEstimator, TransformerMixin):
    """One forest plus one path projection, used as the non-iterative baseline."""

    def __init__(self, estimator, n_components=32, random_state=None):
        self.estimator = estimator
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.forest_ = clone(self.estimator).fit(X, y)
        self.encoder_ = DecisionPathEncoder().fit(self.forest_)
        V = self.encoder_.transform(X)
        self.projector_ = make_projector(
            self.n_components, "gaussian", self.random_state
        ).fit(V)
        return self

    def transform(self, X):
        check_is_fitted(self, ("forest_", "encoder_", "projector_"))
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        return self.projector_.transform(self.encoder_.transform(X))


class NoConcatForestSketch(BaseEstimator, TransformerMixin):
    """Iteration ablation that replaces X with Z instead of concatenating."""

    def __init__(self, estimator, n_components=32, n_iterations=1, random_state=None):
        self.estimator = estimator
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        seeds = np.random.RandomState(self.random_state).randint(
            0, np.iinfo(np.int32).max, size=self.n_iterations + 1
        )
        self.initial_projector_ = make_projector(
            self.n_components, "sparse", seeds[0]
        ).fit(X)
        current = self.initial_projector_.transform(X)
        self.encoders_ = []
        self.projectors_ = []
        for iteration in range(self.n_iterations):
            forest = clone(self.estimator).fit(current, y)
            encoder = DecisionPathEncoder().fit(forest)
            V = encoder.transform(current)
            projector = make_projector(
                self.n_components, "gaussian", seeds[iteration + 1]
            ).fit(V)
            current = projector.transform(V)
            self.encoders_.append(encoder)
            self.projectors_.append(projector)
        return self

    def transform(self, X):
        check_is_fitted(self, ("initial_projector_", "encoders_"))
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        current = self.initial_projector_.transform(X)
        for encoder, projector in zip(self.encoders_, self.projectors_):
            current = projector.transform(encoder.transform(current))
        return current


class MatchedRandomSparsePathTransformer(BaseEstimator, TransformerMixin):
    """Control with random sparse features matched to the real path matrix shape."""

    def __init__(self, estimator, n_components=32, random_state=None):
        self.estimator = estimator
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]
        self.forest_ = clone(self.estimator).fit(X, y)
        self.encoder_ = DecisionPathEncoder().fit(self.forest_)
        V = self.encoder_.transform(X)
        self.n_path_features_ = V.shape[1]
        self.nnz_per_row_ = max(1, int(round(V.nnz / V.shape[0])))
        random_V = self._random_sparse_features(V.shape[0])
        self.projector_ = make_projector(
            self.n_components, "gaussian", self.random_state
        ).fit(random_V)
        return self

    def _random_sparse_features(self, n_rows):
        rng = np.random.RandomState(self.random_state)
        rows = np.repeat(np.arange(n_rows), self.nnz_per_row_)
        cols = rng.randint(0, self.n_path_features_, size=rows.size)
        signs = rng.choice(np.array([-1.0, 1.0]), size=rows.size)
        return sparse.coo_matrix(
            (signs, (rows, cols)),
            shape=(n_rows, self.n_path_features_),
        ).tocsr()

    def transform(self, X):
        check_is_fitted(self, ("forest_", "encoder_", "projector_"))
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has an unexpected number of features")
        V = self.encoder_.transform(X)
        random_V = self._random_sparse_features(V.shape[0])
        return self.projector_.transform(random_V)


def full_sketch(seed, n_components=32, n_iterations=2, kind="classifier"):
    estimator = (
        forest_classifier(seed) if kind == "classifier" else forest_regressor(seed)
    )
    return ForestSketchEstimator(
        estimator=estimator,
        n_components=n_components,
        n_iterations=n_iterations,
        random_state=seed,
    )


forest_sketch = full_sketch


def timed_fit_transform(transformer, X, y):
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    transformed = transformer.fit_transform(X, y)
    return transformed, time.perf_counter() - start_wall, time.process_time() - start_cpu


def serialized_size_mb(value):
    return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)) / 2**20


def classification_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return float(accuracy_score(y_test, prediction)), int(np.count_nonzero(prediction != y_test))


def regression_scores(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return float(r2_score(y_test, prediction)), float(mean_squared_error(y_test, prediction) ** 0.5)


def print_verdict(name, supported, evidence):
    label = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"{name}: {label} — {evidence}")


def critical_difference_stats(
    results,
    block_column,
    method_column,
    score_column="accuracy",
    higher_is_better=True,
):
    """Return paired scores, average ranks, and Nemenyi p-values for a CD plot."""
    import scikit_posthocs as sp

    wide = results.pivot(index=block_column, columns=method_column, values=score_column)
    if wide.isna().any().any():
        raise ValueError("critical-difference input must contain every method for every block")
    average_ranks = wide.rank(
        axis=1,
        ascending=not higher_is_better,
        method="average",
    ).mean()
    significance = sp.posthoc_nemenyi_friedman(wide)
    return wide, average_ranks, significance


def plot_critical_difference(
    results,
    block_column,
    method_column,
    score_column="accuracy",
    higher_is_better=True,
    ax=None,
    title=None,
):
    """Plot a critical-difference diagram from paired experiment results."""
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    _, average_ranks, significance = critical_difference_stats(
        results,
        block_column=block_column,
        method_column=method_column,
        score_column=score_column,
        higher_is_better=higher_is_better,
    )
    sp.critical_difference_diagram(
        average_ranks.to_dict(),
        significance,
        alpha=0.05,
        ax=ax,
        left_only=True,
    )
    if title is not None:
        ax.set_title(title)
    return ax
