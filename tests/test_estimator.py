import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from forestsketch import ForestSketchEstimator, SignedHashProjector


def test_classifier_transform():
    X, y = make_classification(
        n_samples=48,
        n_features=8,
        n_informative=5,
        random_state=1,
    )
    forest = RandomForestClassifier(
        n_estimators=8,
        max_depth=4,
        random_state=11,
    )
    sketch = ForestSketchEstimator(
        estimator=forest,
        n_components=5,
        n_iterations=2,
        random_state=19,
    )

    output = sketch.fit_transform(X, y)
    transformed = sketch.transform(X[:7])

    assert output.shape == (48, 5)
    assert transformed.shape == (7, 5)
    assert sketch.projection_matrices_["P0"].shape == (5, 8)
    assert sketch.projection_matrices_["PZ"][0].shape[0] == 5
    assert sketch.projection_matrices_["PC"][0].shape == (5, 10)
    assert not hasattr(forest, "estimators_")


def test_regressor_is_supported():
    X, y = make_regression(
        n_samples=30,
        n_features=6,
        n_informative=4,
        random_state=2,
    )
    sketch = ForestSketchEstimator(
        estimator=RandomForestRegressor(n_estimators=5, random_state=3),
        n_components=4,
        n_iterations=1,
        random_state=4,
    )

    output = sketch.fit_transform(X, y)

    assert output.shape == (30, 4)


def test_expanding_dimension_mode_sets_output_width():
    X, y = make_classification(
        n_samples=48,
        n_features=8,
        n_informative=5,
        random_state=1,
    )
    sketch = ForestSketchEstimator(
        estimator=RandomForestClassifier(n_estimators=8, max_depth=4, random_state=11),
        n_components=5,
        n_iterations=2,
        dimension_mode="expanding",
        random_state=19,
    )

    output = sketch.fit_transform(X, y)

    assert output.shape == (48, 15)
    assert sketch.projection_matrices_["PC"] == [None, None]
    assert len(sketch.get_feature_names_out()) == 15
    assert sketch.output_dimension_ == 15


def test_sample_weight_and_output_format_are_supported():
    X, y = make_classification(n_samples=30, n_features=6, random_state=12)
    sketch = ForestSketchEstimator(
        estimator=RandomForestClassifier(n_estimators=4, random_state=13),
        n_components=4,
        n_iterations=1,
        output_format="sparse",
        random_state=14,
    )

    output = sketch.fit_transform(X, y, sample_weight=np.ones(X.shape[0]))

    assert sparse.issparse(output)
    transformed = sketch.transform(X)
    assert sparse.issparse(transformed)
    np.testing.assert_allclose(output.toarray(), transformed.toarray())
    assert not hasattr(sketch, "_training_output_")


def test_expanding_mode_applies_concat_normalization():
    X, y = make_classification(n_samples=36, n_features=6, random_state=15)
    sketch = ForestSketchEstimator(
        estimator=RandomForestClassifier(n_estimators=4, random_state=16),
        n_components=4,
        n_iterations=1,
        dimension_mode="expanding",
        normalization="l2",
        random_state=17,
    )

    sketch.fit(X, y)
    output = sketch.transform(X[:5])
    norms = np.sqrt(np.asarray(output ** 2).sum(axis=1))

    assert np.allclose(norms, 1.0)
    assert sketch.concat_normalizers_[0] is not None


def test_signed_hash_projector_is_deterministic_and_sparse():
    X = sparse.csr_matrix(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
        ]
    )
    first = SignedHashProjector(n_components=3, random_state=7).fit(X)
    second = SignedHashProjector(n_components=3, random_state=7).fit(X)

    first_output = first.transform(X)
    second_output = second.transform(X)

    assert sparse.issparse(first_output)
    np.testing.assert_allclose(first_output.toarray(), second_output.toarray())


def test_l2_normalization_returns_valid_output():
    X, y = make_classification(
        n_samples=24,
        n_features=5,
        n_informative=3,
        random_state=5,
    )
    sketch = ForestSketchEstimator(
        estimator=RandomForestClassifier(n_estimators=4, random_state=6),
        n_components=3,
        n_iterations=1,
        normalization="l2",
        random_state=8,
    )

    sketch.fit(X, y)
    output = sketch.transform(X[:4])

    assert output.shape == (4, 3)
