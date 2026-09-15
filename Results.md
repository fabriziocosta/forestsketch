# Experimental results

This document compiles the results printed by the executed hypothesis notebooks. These are exploratory studies, not final benchmark claims. Unless noted otherwise, accuracy is held-out classification accuracy and errors are integer misclassification counts. The notebooks retain their mean and standard-deviation summaries and, where at least three configurations are compared across paired seeds, also provide critical-difference diagrams based on average ranks and Nemenyi post-hoc significance testing.

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
| Dimension scaling relationship | `14_dimension_scaling_relationship.ipynb` | Exploratory 100/500-tree p/m/d study: pooled accuracy association was strongest for d/p (Spearman ρ=0.593), then d/m (0.561), then absolute d (0.478); within-block signs were positive for both ratios in all eight blocks, but thresholds varied widely. |

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

## Dimension scaling relationship — 100/500-tree follow-up

The executed `14_dimension_scaling_relationship.ipynb` study used three controlled synthetic classification datasets with p=12, 48, and 120, the scikit-learn breast-cancer dataset with p=30, forests with exactly 100 and 500 trees, six target dimensions (8, 32, 128, 512, 2048, and 8192), and two paired seeds. Trees used `max_depth=8` and `n_jobs=-1`; the sweep used sparse path projection and non-materialized signed-hash concatenation. Every configuration recorded p, sample count, fitted path width m, d/p, d/m, raw-forest accuracy, Forest Sketch accuracy, separate fit/transform/evaluation wall times, an output-memory proxy, serialized estimator size, and status. All 96 configurations completed successfully in 75.4 seconds.

The planning pass showed the expected growth in raw fitted path width. For the synthetic datasets, the planning m was about 9.9–10.1k columns with 100 trees and 49.8–49.9k with 500 trees; the breast-cancer reference ranged from 2,818 to 14,088 columns. At d=8192 and 500 trees, a dense Gaussian path matrix was estimated at about 3.1 GiB for the synthetic datasets and 880.5 MiB for breast cancer, while the sparse estimates were about 14 MiB and 7.4 MiB respectively. The sparse implementation made the requested full grid tractable; these are planning estimates, not peak-RSS measurements.

Across all successful rows, pooled Spearman correlations between accuracy and log-width were 0.478 for absolute d, 0.593 for d/p, and 0.561 for d/m. The best observed configuration in each dataset/forest block was:

| Dataset | Trees | Best d | Mean m | Mean sketch accuracy | Mean raw-RF accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Breast cancer (p=30) | 100 | 2048 | 2112 | 0.965 | 0.947 |
| Breast cancer (p=30) | 500 | 128 | 13510 | 0.971 | 0.947 |
| Synthetic (p=12) | 100 | 8192 | 4894 | 0.904 | 0.907 |
| Synthetic (p=12) | 500 | 512 | 30304 | 0.904 | 0.900 |
| Synthetic (p=120) | 100 | 8192 | 8012 | 0.856 | 0.759 |
| Synthetic (p=120) | 500 | 8192 | 40008 | 0.859 | 0.815 |
| Synthetic (p=48) | 100 | 8192 | 6958 | 0.889 | 0.867 |
| Synthetic (p=48) | 500 | 8192 | 34946 | 0.907 | 0.856 |

### Within-block analysis

The grouped analysis removes the pooled factor mixing by analyzing each dataset × forest-size block separately. Each block contains 12 rows (six dimensions × two seeds). The accuracy-gap correlations were positive for both ratios in all eight blocks:

| Dataset | Trees | ρ(log(d/p)) | ρ(log(d/m)) | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| Breast cancer (p=30) | 100 | 0.607 | 0.505 | -0.006 |
| Breast cancer (p=30) | 500 | 0.557 | 0.484 | -0.008 |
| Synthetic (p=12) | 100 | 0.800 | 0.841 | -0.061 |
| Synthetic (p=12) | 500 | 0.738 | 0.807 | -0.062 |
| Synthetic (p=120) | 100 | 0.975 | 0.972 | -0.032 |
| Synthetic (p=120) | 500 | 0.947 | 0.979 | -0.058 |
| Synthetic (p=48) | 100 | 0.954 | 0.923 | -0.052 |
| Synthetic (p=48) | 500 | 0.975 | 0.958 | -0.049 |

The smallest observed ratios whose block-mean gap was at least -0.02 were:

| Dataset | Trees | Minimum d/p (d) | Minimum d/m (d) |
| --- | ---: | ---: | ---: |
| Breast cancer (p=30) | 100 | 1.067 (32) | 0.010 (32) |
| Breast cancer (p=30) | 500 | 4.267 (128) | 0.010 (128) |
| Synthetic (p=12) | 100 | 170.667 (2048) | 0.325 (2048) |
| Synthetic (p=12) | 500 | 42.667 (512) | 0.015 (512) |
| Synthetic (p=120) | 100 | 4.267 (512) | 0.055 (512) |
| Synthetic (p=120) | 500 | 4.267 (512) | 0.011 (512) |
| Synthetic (p=48) | 100 | 42.667 (2048) | 0.266 (2048) |
| Synthetic (p=48) | 500 | 42.667 (2048) | 0.053 (2048) |

Thus the apparent d/p relationship survives as consistent descriptive within-block evidence in this grid, but its threshold varies from 1.067 to 682.667 and does not define a stable universal rule. d/m is also consistently increasing, with thresholds from 0.010 to 0.325. Limitations remain two seeds, controlled datasets, one downstream learner, bounded tree depth, and exploratory use of held-out test results. Larger real datasets and validation-based dimension selection are needed before changing the other hypothesis notebooks.
