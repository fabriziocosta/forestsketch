# Experimental results

This document compiles the results printed by the executed hypothesis notebooks. These are exploratory studies, not final benchmark claims. Unless noted otherwise, accuracy is held-out classification accuracy and errors are integer misclassification counts.

## Summary

| Study | Notebook | Result |
| --- | --- | --- |
| UMAP representation | `03_umap_representations.ipynb` | Visualization study; not a numbered hypothesis. |
| Question 7: target dimension | `04_q7_hypothesis_target_dimension.ipynb` | A useful plateau was reached by d=2048; the fitted power-law asymptote was 0.962. |
| Question 8: expanding dimensionality | `05_q8_hypothesis_expanding_dimension.ipynb` | Inconclusive: expansion beat narrow fixed mode by 0.043, but lost 0.008 against the width-matched fixed control. |
| Question 4: concatenation | `06_q4_hypothesis_concatenation.ipynb` | Technically supported by a 0.001 gain, but the practical effect was negligible. |
| Question 3: iteration | `07_q3_hypothesis_iteration.ipynb` | Not supported: T=2 was worse than T=1 by 0.023, with 95% CI [-0.037, -0.010]. |
| Preliminary initial projection | `08_preliminary_hypothesis_initial_projection.ipynb` | Not supported under the predefined 0.05 accuracy-loss tolerance; the largest observed loss was 0.055. |
| Question 1: predictive performance | `09_q1_hypothesis_predictive_performance.ipynb` | Not supported in the tested classification and regression runs. |
| Question 2: tree-path value | `10_q2_hypothesis_tree_path_value.ipynb` | Supported: real tree paths outperformed the matched random sparse control. |
| Question 5: sample efficiency | `11_q5_hypothesis_sample_efficiency.ipynb` | Not supported: the low-data Forest Sketch advantage was -0.008. |
| Question 6: quality-cost trade-off | `12_q6_hypothesis_cost_tradeoff.ipynb` | Not supported in the tested cost envelope; accuracy was 0.016 below the original baseline. |

## Preliminary question — initial projection

The notebook tested whether the initial projection preserved random-forest performance within an absolute accuracy-loss tolerance of 0.05. Across two seeds and dimensions 16, 32, and 64, the largest observed loss was 0.055. The predefined criterion was therefore not met.

## Question 1 — predictive performance

Mean classification accuracy was:

| Method | Accuracy | Standard deviation |
| --- | ---: | ---: |
| Original | 0.650 | 0.042 |
| Initial random projection | 0.648 | 0.051 |
| One-shot tree-path | 0.722 | 0.085 |
| Forest Sketch | 0.635 | 0.043 |

Mean regression results were:

| Method | R² | RMSE |
| --- | ---: | ---: |
| Original | 0.993 | 18.208 |
| Initial random projection | 0.464 | 161.986 |
| One-shot tree-path | 0.292 | 185.450 |
| Forest Sketch | 0.117 | 205.560 |

The full Forest Sketch method did not improve over the original representation or the initial projection in these runs, so the hypothesis was not supported.

## Question 2 — tree-path representation value

Mean classification accuracy was 0.722 for the one-shot real tree-path representation and 0.507 for the matched random sparse control. Forest Sketch reached 0.635, while the initial random projection reached 0.648. The real path features beat the random sparse control, supporting the narrower claim that the forest paths carry useful signal.

## Question 3 — iteration

The fixed-dataset paired experiment used 12 model and projection seeds:

| Iterations | Mean accuracy | Standard deviation | 95% CI half-width |
| ---: | ---: | ---: | ---: |
| 0 | 0.682 | 0.014 | 0.008 |
| 1 | 0.692 | 0.016 | 0.009 |
| 2 | 0.669 | 0.025 | 0.014 |
| 3 | 0.655 | 0.027 | 0.015 |

The paired T=2 minus T=1 gain was -0.023, with 95% CI [-0.037, -0.010]. The second iteration was consistently worse in this experiment. Total wall-clock time was 52.9 seconds with the fully parallel experiment forest configuration.

## Question 4 — concatenation

Mean accuracy was 0.635 with full concatenation and 0.634 with replacement without concatenation. The measured gain was only 0.001, so the notebook labels the hypothesis supported technically but practically negligible.

## Question 5 — sample efficiency and robustness

Across the tested fractions, mean accuracy was 0.634 for Forest Sketch and 0.653 for the original representation. At 25% of the training data, the Forest Sketch advantage was -0.008. The hypothesis was not supported in this exploratory study.

## Question 6 — quality-cost trade-off

| Method | Accuracy | Wall time (s) | CPU time (s) | Peak Python memory (MB) | Serialized state (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.620 | 1.512 | 5.108 | 1.445 | 0.005 |
| Initial random projection | 0.611 | 3.093 | 10.359 | 0.737 | 0.007 |
| Forest Sketch | 0.605 | 5.403 | 14.997 | 30.406 | 6.305 |

Forest Sketch was slower and used more memory while scoring below the original baseline by 0.016. The hypothesis was not supported under this cost envelope.

## Question 7 — target dimension

The fixed 120-feature experiment swept target dimensions from 8 through 8192, multiplying by four between measurements. The raw random forest baseline reached 0.843 accuracy, with an uncompressed visited-node representation of 7,754 columns, indexed from 0 through 7,753.

| d | Mean accuracy | Standard deviation | Output MB | Serialized model MB |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.592 | 0.026 | 0.081 | 2.794 |
| 32 | 0.660 | 0.015 | 0.322 | 6.241 |
| 128 | 0.728 | 0.018 | 1.289 | 17.059 |
| 512 | 0.807 | 0.010 | 5.156 | 56.273 |
| 2048 | 0.835 | 0.008 | 20.625 | 193.447 |
| 8192 | 0.848 | 0.013 | 82.500 | 690.940 |

The best observed mean was 0.848 at d=8192. The smallest dimension within 0.02 of that best was d=2048. The saturating power-law fit $A - B/d^\\alpha$ estimated $A=0.962$, $B=0.555$, and $\\alpha=0.187$. Total wall-clock time was 37.3 seconds. Dimensions above 120 are valid but are no longer dimensionality-reducing relative to the original input.

## Question 8 — expanding dimensionality

This experiment used eight paired seeds on one fixed classification split, with an initial block dimension of 16 and two iterations:

| Mode | Output width | Mean accuracy | Standard deviation | Mean serialized state (MB) |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 16 | 0.657 | 0.019 | 1.473 |
| Expanding | 48 | 0.701 | 0.026 | 1.459 |
| Width-matched fixed | 48 | 0.708 | 0.041 | 3.108 |

Expanding minus narrow fixed mode was +0.043, with 95% CI [0.021, 0.066]. Against the width-matched fixed control, the gain was -0.008, with 95% CI [-0.041, 0.026]. The evidence is therefore inconclusive, and the apparent gain is plausibly attributable to the larger representation width rather than expansion itself. Total wall-clock time was 53.1 seconds.

## Overall interpretation

The experiments provide evidence that real tree-path features contain signal beyond a matched random sparse control. They do not yet establish that the complete iterative Forest Sketch architecture improves predictive performance over simpler baselines. In particular, later fixed-width iterations were harmful in the current study, and expanding dimensionality did not beat a width-matched fixed representation.
