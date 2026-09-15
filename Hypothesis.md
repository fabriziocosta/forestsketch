# Scientific Hypotheses

## Aim

The goal is to determine whether Forest Sketch produces a useful representation beyond simply using the original features or applying a one-time random projection. The experiments should establish both predictive value and the cost of obtaining that value.

The method should not be considered universally superior from a single successful result. Evidence should come from repeated comparisons against strong baselines, ablations, and held-out data.

## Experimental principles

- Use fixed train, validation, and test splits for every method on a given task.
- Repeat each experiment across multiple random seeds and report the mean and uncertainty.
- Match the target representation dimension d across methods.
- Keep the downstream estimator, tuning budget, and preprocessing policy comparable.
- Record wall-clock time, peak memory, model size, and representation size in addition to predictive metrics.
- Select hyperparameters using validation data only; reserve the test set for the final comparison.

## Baselines and ablations

| Method | Purpose |
| --- | --- |
| Original features X | Measures performance without the proposed representation |
| Initial random projection X₀ | Measures the value of dimensionality reduction alone |
| One-shot tree-path projection | Measures the value of forest node-path features without iteration |
| Iterative method with no concatenation | Tests whether retaining the previous representation is important |
| Iterative method with shuffled path features | Tests whether gains come from meaningful tree structure rather than extra random features |
| Full Forest Sketch | Proposed method |

All baselines should use comparable downstream models and, where possible, comparable feature and compute budgets.

## Preliminary question: Does the initial projection preserve forest performance?

**Hypothesis.** Applying the initial random projection to obtain X₀ does not materially reduce the performance of a random forest compared with fitting the same forest directly on the original representation X.

Fit matched random-forest models on X and X₀, using the same train and test splits, forest configuration, tuning budget, and random seeds. Report the performance difference across several target dimensions d and datasets. Define a practical tolerance before evaluating the results, such as an absolute metric loss small enough that the later tree-path and iterative gains could reasonably compensate for it.

**Evidence for the hypothesis:** performance on X₀ remains within the predefined tolerance of performance on X across seeds and datasets, or any small loss is offset by the later tree-path representation.

**Falsification:** the initial projection causes a large or inconsistent performance loss. In that case, test a larger d, a different projection family, or a forest trained on X with projection applied only to the extracted path representation.

## Question 1: Does the representation improve predictive performance?

**Hypothesis.** The full method improves test-set performance over the original features and over the initial random projection at the same target dimension.

Evaluate this on several supervised datasets and task types. Use an appropriate primary metric for each task, such as accuracy or macro-F1 for classification and RMSE or R² for regression.

**Evidence for the hypothesis:** a consistent improvement across datasets and seeds, with uncertainty intervals that support a practically meaningful gain.

**Falsification:** performance is no better than the original-feature baseline, or gains occur only on one dataset or one seed.

## Question 2: Does the tree-path representation add value beyond random projection?

**Hypothesis.** Projected node-path features improve performance over applying a random projection directly to X.

Compare the initial random projection, the one-shot tree-path projection, and the full iterative method. Include a control in which the path-feature values are replaced by random sparse features with the same shape and sparsity.

**Evidence for the hypothesis:** the real tree-path representation outperforms both direct random projection and the matched random-feature control.

**Falsification:** matched random features perform equally well, suggesting that the benefit may come from dimensionality, sparsity, or additional model capacity rather than tree structure.

## Question 3: Does iteration provide value over a one-shot transformation?

**Hypothesis.** Reusing the compressed representation as the input to later forests improves performance over a single forest and projection.

Measure performance as the number of iterations T increases, for example T = 0, 1, 2, 3, and a larger selected value. Plot performance against computation and memory.

**Evidence for the hypothesis:** an improvement from T = 1 to a small number of later iterations that is reproducible and does not depend on test-set tuning.

**Falsification:** additional iterations do not improve performance, or they improve training performance while harming held-out performance.

## Question 4: Is concatenating the previous representation important?

**Hypothesis.** The update using Cₜ, which combines the previous representation with the newly projected forest representation, is better than replacing the previous representation entirely.

Compare the full update with a replacement update that uses only Zₜ, and with a control that concatenates but does not apply the second projection.

**Evidence for the hypothesis:** the full update retains useful information and gives better validation and test performance at the same final dimension.

**Falsification:** replacement performs as well or better, indicating that the residual information from the previous iteration is unnecessary.

## Question 5: Does the method improve sample efficiency and robustness?

**Hypothesis.** The representation provides an advantage when labeled training data are limited and remains competitive under moderate perturbations or distribution shift.

Construct learning curves using progressively larger fractions of the training set. Where appropriate, test robustness to feature noise, missing values, or a temporally or geographically separated test set.

**Evidence for the hypothesis:** the proposed representation reaches a target performance level with fewer labeled samples and degrades no faster than the baselines under the chosen perturbations.

**Falsification:** the method only helps with abundant data or is substantially more fragile than the original-feature baseline.

## Question 6: What is the quality-cost trade-off?

**Hypothesis.** Any performance gain is large enough to justify the additional forest fitting, sparse path extraction, projection storage, and inference cost.

Report:

- predictive performance;
- fit and transform time;
- peak memory;
- serialized model size;
- number of trees and total forest nodes;
- target dimension d;
- number of iterations T.

Compare methods at matched performance and at matched compute budgets.

**Evidence for the hypothesis:** the full method gives a meaningful accuracy or generalization benefit within an acceptable cost envelope.

**Falsification:** improvements are negligible relative to the added time, memory, or model size.

## Question 7: How sensitive is the method to the target dimension?

**Hypothesis.** There is a range of target dimensions in which the representation preserves the useful tree structure without requiring the full sparse path dimension.

Evaluate a logarithmic grid of target dimensions and compare performance, storage, and runtime. Report whether the best dimension is stable across random seeds and datasets.

**Evidence for the hypothesis:** performance reaches a plateau at a substantially smaller dimension than the uncompressed path representation.

**Falsification:** performance is highly unstable or requires a dimension so large that the compression is not useful.

## Question 8: Is expanding dimensionality better than fixed-dimensional re-embedding?

**Hypothesis.** Retaining each projected tree representation through concatenation, so that the representation grows by d features per iteration, improves held-out predictive performance relative to projecting every concatenation back to d features.

**Experiment.** On identical train, validation, and test splits, compare `dimension_mode="fixed"` and `dimension_mode="expanding"` at the same initial block dimension d and iteration count T. Use the same supplied forest configuration, projection seeds, downstream estimator, and repeated model seeds. Report both predictive performance and final dimensionality, and include a width-matched fixed-dimensional control where practical.

**Measure.** The primary measure is the paired held-out accuracy difference between expanding and fixed modes. Secondary measures are representation width, wall-clock fit time, and serialized fitted-estimator size. A paired confidence interval across seeds is required before calling the hypothesis supported.

**Evidence for the hypothesis:** expanding mode has a reproducible positive held-out gain that remains after accounting for its larger representation, or achieves the same performance with a smaller initial block dimension.

**Falsification:** expanding mode does not improve held-out performance, its gain disappears under width-matched comparison, or its added width and cost are not justified by the gain.

## Recommended primary claim

The strongest initial claim would be:

> At a fixed representation dimension and comparable downstream model budget, Forest Sketch improves held-out predictive performance over original features, direct random projection, and a one-shot tree-path projection, with a cost that is justified by the improvement.

This claim is supported only if Questions 1–3 show consistent gains and Question 6 shows that the gains are not merely an artifact of an impractical computational budget.

## Minimum first experiment

Start with one classification and one regression dataset:

1. Fit the original-feature, initial-random-projection, one-shot, and full iterative baselines.
2. Use the same downstream estimator and target dimensions.
3. Repeat each configuration over several random seeds.
4. Sweep a small number of iteration counts.
5. Report test performance with uncertainty, fit time, transform time, peak memory, and serialized model size.

This experiment will reveal whether the idea is promising enough to justify broader robustness and ablation studies.
