# Forest Sketch

Forest Sketch is an iterative representation-learning method that turns random-forest decision paths into compact feature vectors.

For each sample, the method records the internal decision nodes and terminal leaves visited by the trees in a forest. This creates a wide sparse representation that is then compressed with a random projection. The compressed tree representation is concatenated with the current input representation and projected back to a fixed user-defined dimension. The process can be repeated for several iterations.

The project is intended to provide a scikit-learn-compatible transformer named ForestSketchEstimator. The transformer accepts a configured RandomForestClassifier or RandomForestRegressor through its estimator parameter and uses that forest internally at every iteration.

## Status

This repository contains an initial working implementation, together with the architecture and experimental hypotheses that guide it. The API below describes the current first usable interface.

## Intended usage

The estimator is a transformer: it learns a compact representation and can be placed inside a scikit-learn Pipeline with a downstream classifier or regressor. The internal forest estimator and the downstream estimator are configured independently.

~~~python
from forestsketch import ForestSketchEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=7,
)

model = make_pipeline(
    ForestSketchEstimator(
        estimator=forest,
        n_components=64,
        n_iterations=3,
        random_state=42,
    ),
    LogisticRegression(max_iter=2000),
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
~~~

The supplied forest controls the tree-path features generated inside Forest Sketch. The downstream estimator is deliberately separate. This makes it possible to configure the internal forest independently and evaluate the resulting representation with different scikit-learn models.

## Direct transformation

The transformer should also support the standard scikit-learn lifecycle:

~~~python
sketch = ForestSketchEstimator(
    estimator=RandomForestClassifier(
        n_estimators=100,
        random_state=7,
    ),
    n_components=64,
    n_iterations=3,
    random_state=42,
)

X_train_sketch = sketch.fit_transform(X_train, y_train)
X_test_sketch = sketch.transform(X_test)
~~~

The output of both calls should have shape n_samples × n_components.

For experiments and diagnostics, the estimator should provide an opt-in method that returns the final representation and the per-iteration trace:

~~~python
X_test_sketch, trace = sketch.transform_with_intermediates(X_test)

first_iteration = trace[0]
V_0 = first_iteration["V"]
Z_0 = first_iteration["Z"]
C_0 = first_iteration["C"]
X_1 = first_iteration["X_next"]
~~~

The standard transform method should not retain these intermediate values by default, because the sparse visited-node matrices can be very large.

## Initial API

The first implementation is expected to expose:

| Parameter | Meaning |
| --- | --- |
| estimator | Configured RandomForestClassifier or RandomForestRegressor used internally |
| n_components | Target dimension of the compact representation |
| n_iterations | Number of forest, path-extraction, and projection cycles |
| random_state | Seed for projection generation; the forest seed is configured on estimator |
| initial_projection_type | Projection family for the original input projection |
| path_projection_type | Projection family for sparse forest-path features |
| concat_projection_type | Projection family for the concatenation projection |
| normalization | No normalization, row-wise L1 normalization, or row-wise L2 normalization |
| path_encoder | Optional replacement for the decision-path encoder |
| normalizer | Optional replacement for the default normalization component |

The internal forest should be cloned before fitting so that ForestSketchEstimator does not mutate the estimator object supplied by the caller. Its forest hyperparameters, including tree count, depth, feature subsampling, and forest random seed, are configured directly on estimator. The projection parameters remain ForestSketchEstimator parameters.

The exact parameter names may be refined to follow scikit-learn conventions, but the estimator should support BaseEstimator and TransformerMixin behavior.

## Modular components

ForestSketchEstimator should be composed from replaceable components with standard scikit-learn-style interfaces:

- the supplied forest estimator;
- a path encoder for visited internal and leaf nodes;
- a normalizer, including an identity normalizer for the no-normalization condition;
- a projector, including materialized and deterministic signed-hashing implementations;
- the iteration coordinator.

This separation is intended to make experiments comparable. A change to the projector or normalizer should not require changing the forest estimator or the downstream model.

For sparse path features, row-wise L1 or L2 normalization rescales existing nonzero entries and preserves sparsity. Mean-centering should be avoided by default because it generally densifies the matrix.

## Algorithm

Given an input matrix X:

~~~text
X₀ = initial_random_projection(X)
X_current = X₀

for each iteration t:
    forest_t = clone_and_fit(estimator, X_current, y)
    Vₜ = collect_all_visited_internal_and_leaf_nodes(forest_t, X_current)
    Zₜ = Vₜ projected to n_components
    Cₜ = concatenate(X_current, Zₜ)
    X_current = Cₜ projected back to n_components

return X_current
~~~

The initial projection ensures that the first forest receives the configured representation dimension. At every later iteration, the forest input and output remain compact even though the intermediate visited-node matrix may be very wide.

## Learned state and reproducibility

Forest Sketch should materialize and retain all learned components during fit:

- the initial projection matrix;
- the cloned forest fitted at each iteration;
- the path-feature projection matrix for each iteration;
- the concatenation projection matrix for each iteration;
- the global node-column layout for each forest.

The same state must be reused by transform. A random seed is useful for reproducibility metadata, but inference should not regenerate projection matrices from the seed.

## Comparison baselines

Initial experiments should compare Forest Sketch with:

1. A downstream model trained directly on the original features.
2. An initial random projection followed by the same downstream model.
3. A one-shot random-forest path projection without iteration.
4. An ablation that removes the concatenation with the previous representation.
5. A matched random sparse-feature control.

Use the same train/test splits, target dimension, downstream model, and tuning budget wherever possible.

## Versioning and releases

Forest Sketch follows semantic versioning with `MAJOR.MINOR.PATCH` versions. Releases are automated from pushes to the `main` branch using Conventional Commits:

- `fix:` and maintenance commits produce a patch release, such as `0.1.1`.
- `feat:` produces a minor release, such as `0.2.0`.
- A breaking change, marked with `!` or a `BREAKING CHANGE:` footer, produces a major release, such as `1.0.0`.

The release workflow updates the version in `pyproject.toml`, generates the changelog, creates a Git tag, and publishes a GitHub Release. The initial public baseline is version `0.1.0`. PyPI publishing is intentionally not enabled yet; the GitHub repository and its release artifacts are the current distribution target.

When contributing, use a Conventional Commit message, for example:

~~~text
feat: add a new projection backend
fix: preserve sparse path features during normalization
~~~

## Design documents

- architecture.md describes the representation flow, matrix dimensions, and scikit-learn integration.
- Hypothesis.md lists the scientific questions, baselines, ablations, and falsification criteria.
