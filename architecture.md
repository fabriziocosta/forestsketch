# Forest Sketch

## Overview

Forest Sketch is an iterative representation-learning architecture built from two operations:

1. A random forest converts an input representation into high-dimensional sparse features that describe the nodes visited by each tree.
2. A random projection compresses those sparse features to a user-defined dimension.

The compressed tree representation is concatenated with the representation entering the current iteration and projected back to the same fixed dimension. The result becomes the input to the next random-forest stage.

## Internal forest estimator

ForestSketchEstimator accepts a configured RandomForestClassifier or RandomForestRegressor through an estimator parameter. The supplied forest is a template for the internal forest stages; its tree count, depth, feature subsampling, fitting options, and forest random seed are configured independently from the projection parameters.

At every iteration, the implementation should clone the supplied estimator, fit the clone on the current representation Xₜ and target y, and use that fitted clone to produce Vₜ. Cloning prevents the caller's estimator object from being mutated and allows each iteration to retain its own fitted forest.

The Forest Sketch projection settings control the target dimension, projection family, projection seeds, and number of iterations. They do not replace or silently override the configuration of the supplied forest.

## Modular implementation

ForestSketchEstimator should be an orchestrator over replaceable components with scikit-learn-style interfaces. Each component should expose fit, transform, and fit_transform behavior where appropriate, and should be compatible with cloning and parameter inspection.

The initial component boundaries are:

- forest estimator: a configured RandomForestClassifier or RandomForestRegressor supplied by the caller;
- path encoder: converts a fitted forest and samples into the sparse visited-node matrix Vₜ;
- normalizer: leaves data unchanged or applies a documented normalization policy;
- projector: materializes or deterministically generates a projection from an input dimension to d;
- iteration coordinator: concatenates representations and applies the stage-specific projector.

ForestSketchEstimator should compose these components rather than hard-code one implementation. This makes it possible to compare materialized random projections, on-the-fly signed hashing, different normalizers, and different path encoders under the same estimator interface.

## Notation

- X is the original input matrix.
- Xₜ is the representation entering iteration t.
- Vₜ is the sparse visited-node feature matrix produced by the forest.
- Zₜ is the compressed tree representation.
- Cₜ is the concatenated representation.
- P is reserved for every random-projection matrix.

The original input has p features. The configured representation dimension is d. The initial representation is obtained by projecting X:

$$
X_0 = X P_0, \qquad P_0 \in \mathbb{R}^{p \times d}
$$

Thus, X₀ is the first forest input and has dimension d.

## Tree-path representation

At iteration t, a clone of the supplied scikit-learn random forest estimator is fitted using Xₜ. For each sample and each tree, collect every node visited on the path from the root to the terminal leaf. Both internal decision nodes and the leaf node are included.

Each distinct forest node is treated as a feature. The resulting matrix Vₜ is very wide and sparse:

$$
V_t \in \mathbb{R}^{n \times m_t}
$$

where n is the number of samples and mₜ is the number of globally indexed nodes in the forest at iteration t.

Conceptually:

~~~text
Xₜ ──► random forest ──► visited internal and leaf nodes ──► Vₜ
~~~

Forest-local node indices must be made globally unique before combining paths from multiple trees. A disjoint column range for each tree is one possible implementation.

## Random projections

The sparse path matrix is compressed to the configured dimension:

$$
Z_t = V_t P_t^Z, \qquad P_t^Z \in \mathbb{R}^{m_t \times d}
$$

The matrix Pₜᶻ is sampled once during fitting and reused for all transformations at iteration t.

The initial projection and the tree-path projection are not the same matrix:

$$
P_0 \in \mathbb{R}^{p \times d},
\qquad
P_t^Z \in \mathbb{R}^{m_t \times d}
$$

The first depends on the original feature count p; the second depends on the number of forest nodes mₜ.

## Normalization

Normalization should be an explicit, configurable component rather than an implicit side effect of projection. The simplest public option is no normalization versus normalization, with the implementation also allowing a specific policy such as row-wise L1 or L2 normalization.

For the sparse visited-node matrix Vₜ, row-wise normalization is sparse-safe: it rescales the existing nonzero indicators without changing their locations. An L1 policy makes the total path mass comparable across samples, while an L2 policy controls the Euclidean norm. The implementation must define how zero-norm rows are handled; the safe default is to leave them unchanged.

Mean-centering sparse data must not be used by default because subtracting a feature mean generally turns the sparse matrix into a dense matrix. A standard sparse-aware row normalizer is appropriate; a centering-based scaler should require an explicit opt-in.

Normalization may be applied independently before each projection:

- before P₀, to the original input X;
- before Pₜᶻ, to the sparse path matrix Vₜ;
- before Pₜᶜ, to the concatenated representation Cₜ.

For experiments, expose these choices through replaceable normalizer components or stage-specific settings. The initial baseline should use no normalization, with row-wise L2 normalization as the first comparison.

## Iterative update

The compressed tree representation is concatenated with the current forest input:

$$
C_t = [X_t \; || \; Z_t],
\qquad
C_t \in \mathbb{R}^{n \times 2d}
$$

A third projection maps the concatenated representation back to dimension d:

$$
X_{t+1} = C_t P_t^C,
\qquad
P_t^C \in \mathbb{R}^{2d \times d}
$$

The complete iteration is:

~~~text
Xₜ
  │
  ├──► fit random forest
  │       │
  │       └──► collect visited-node features: Vₜ
  │                         │
  │                         └──► project with Pₜᶻ: Zₜ
  │
  └──────────── concatenate [Xₜ || Zₜ]: Cₜ
                              │
                              └──► project with Pₜᶜ: Xₜ₊₁
~~~

The next forest operates on Xₜ₊₁. After T iterations, the final representation is the current X representation at t = T.

## Intermediate representations

The API should expose intermediate representations for inspection and diagnostics, but capture should be opt-in because Vₜ can be extremely wide and sparse.

The standard transform method should return only the final representation Xₜ. A separate method such as transform_with_intermediates should return the final representation together with a per-iteration trace containing:

- Xₜ, the forest input;
- Vₜ, the sparse visited-node representation;
- Zₜ, the projected tree representation;
- Cₜ, the concatenated representation;
- Xₜ₊₁, the next forest input.

This keeps ForestSketchEstimator compatible with ordinary scikit-learn Pipeline behavior while making the internal computation available for experiments and debugging.

## End-to-end algorithm

Given original input X, target dimension d, and number of iterations T:

~~~text
X₀ = project_to_dimension_d(X)
X_current = X₀

for t in 0, ..., T - 1:
    forest_t = clone_and_fit(estimator, X_current, y)
    V_t = collect_visited_internal_and_leaf_nodes(forest_t, X_current)
    Z_t = project_with_P_t^Z(V_t)
    C_t = concatenate(X_current, Z_t)
    X_current = project_with_P_t^C(C_t)

return X_current
~~~

For supervised learning, the forest-fitting step may use the target y. For unsupervised or representation-only use, the forest must be induced with an explicitly defined proxy objective or other forest construction strategy.

## Projection matrix dimensions

With samples stored as rows, the three projection matrices are:

| Matrix | Shape | Maps |
| --- | --- | --- |
| P₀ | p × d | Original features to the initial representation X₀ |
| Pₜᶻ | mₜ × d | Sparse forest-path features Vₜ to tree representation Zₜ |
| Pₜᶜ | 2d × d | Concatenated representation Cₜ to next representation Xₜ₊₁ |

The corresponding data dimensions are:

| Object | Shape | Description |
| --- | --- | --- |
| X | n × p | Original input features |
| X₀ | n × d | Initial representation |
| Xₜ | n × d | Representation entering iteration t |
| Vₜ | n × mₜ | Sparse visited-node features |
| Zₜ | n × d | Compressed tree representation |
| Cₜ | n × 2d | Concatenation of Xₜ and Zₜ |
| Xₜ₊₁ | n × d | Representation entering the next iteration |

In scikit-learn, the stored components_ array uses the transpose orientation: (d, p) for P₀, (d, mₜ) for Pₜᶻ, and (d, 2d) for Pₜᶜ.

## Training and inference

All forests and projection matrices are part of the fitted transformation and must be retained for inference. A production implementation should distinguish between:

- fit: clone and fit the supplied forest estimator, and materialize P₀, Pₜᶻ, and Pₜᶜ;
- transform: apply the stored forests and projection matrices to new samples;
- fit_transform: fit and transform the training data.

The matrices are sampled once during fitting and reused for every batch and every later inference call. The random seed should be stored as provenance and for reproducibility checks, but inference should use the materialized matrices.

At inference time, new samples must use the same global node-column layout and the same projection matrices as the training data. Forest structure is fixed after fitting, so path extraction for new samples is deterministic given the fitted model.

## scikit-learn path extraction

For a fitted scikit-learn forest, each estimator exposes its underlying tree through estimator.tree_. The decision_path method returns a sparse indicator matrix containing the nodes visited by each sample. Since the decision path includes intermediate nodes and the final leaf, it provides the required feature construction directly.

The implementation should preserve the sparse format throughout path extraction and the first projection. Materializing the full node-feature matrix as a dense array may be infeasible for large forests.

## Design considerations

### Reproducibility

Use separate, documented random seeds for forest construction, the initial projection, path projections, and concatenation projections, or derive them deterministically from one top-level seed.

### Sign and scaling

Random projections can produce signed values even though the path indicators are binary. The implementation should document whether each iteration applies centering, normalization, or scaling before concatenation.

### Memory and sparsity

The visited-node matrix can be very wide. Keep it sparse until projection and consider processing samples in batches if the projected output or concatenated matrix is large.

### Number of iterations

Increasing T repeatedly applies a nonlinear feature-expansion-and-compression cycle. More iterations may capture richer interactions, but can also increase computation, amplify noise, or make interpretation harder.

The number of iterations T and target dimension d are downstream hyperparameters. Select them with cross-validation on the training data, using the objective and metric of the downstream task. The final test set must remain untouched until the configuration is fixed.

## Open questions for the implementation

There are currently no unresolved architectural questions recorded here. The next step is implementation and empirical validation.
