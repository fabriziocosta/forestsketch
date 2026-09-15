# Experimental results

This document compiles the results printed by the executed hypothesis notebooks. These are exploratory studies, not final benchmark claims. Unless noted otherwise, accuracy is held-out classification accuracy and errors are integer misclassification counts. The notebooks retain their mean and standard-deviation summaries and, where at least three configurations are compared across paired seeds, also provide critical-difference diagrams based on average ranks and Nemenyi post-hoc significance testing.

## Summary

| Study | Notebook | Result |
| --- | --- | --- |
| Baseline comparison | `01_compare_baselines.ipynb` | Current p=120, d=2400, 100-tree run: one-shot path 0.856 ± 0.018; Recursive Sketch 0.844 ± 0.025; original 0.706 ± 0.054. |
| Scaling benchmarks | `02_scaling_benchmarks.ipynb` | Current 100-tree run computes d=20p: d=320–2560 across p=16–128; wall time 0.708–1.727 s and peak RSS 948–1503 MiB in the feature sweep. |
| UMAP representation | `03_umap_representations.ipynb` | Special 2D UMAP design retained; 100-tree run completed and generated 7 embeddings. |
| Question 7: target dimension | `04_q7_hypothesis_target_dimension.ipynb` | Sweep retained with 100 trees; raw RF 0.841, sketch 0.845 at d=2048 and 0.847 at d=8192; plateau threshold d=2048. |
| Question 8: expanding dimensionality | `05_q8_hypothesis_expanding_dimension.ipynb` | Current p=60, block d=1200: inconclusive; expanding 0.869 ± 0.017 vs fixed 0.860 ± 0.021 and width-matched fixed 0.876 ± 0.012. |
| Question 4: concatenation | `06_q4_hypothesis_concatenation.ipynb` | Current p=120, d=2400: concatenation 0.830 ± 0.029 vs replacement 0.782 ± 0.042; gain 0.048. |
| Question 3: iteration | `07_q3_hypothesis_iteration.ipynb` | Current p=120, d=2400: T=2 was below T=1 by 0.012, 95% CI [-0.018, -0.007]. |
| Preliminary initial projection | `08_preliminary_hypothesis_initial_projection.ipynb` | Current p=120, d=2400: maximum observed loss 0.000; predefined tolerance 0.05 was met. |
| Question 1: predictive performance | `09_q1_hypothesis_predictive_performance.ipynb` | Current classification p=120,d=2400: Recursive Sketch 0.830 vs original 0.650; regression p=80,d=1600: R² 0.439 vs original 0.993. |
| Question 2: tree-path value | `10_q2_hypothesis_tree_path_value.ipynb` | Current p=120,d=2400: one-shot path 0.825, Recursive Sketch 0.830, random sparse control 0.525. |
| Question 5: sample efficiency | `11_q5_hypothesis_sample_efficiency.ipynb` | Current p=120,d=2400: low-data Recursive Sketch advantage 0.143; full-data mean accuracy 0.830 vs original 0.650. |
| Question 6: quality-cost trade-off | `12_q6_hypothesis_cost_tradeoff.ipynb` | Current p=120,d=2400: Recursive Sketch 0.809 vs original 0.620, with 4.267 s wall time and 1054.5 MiB peak Python memory. |
| Dimension scaling relationship | `14_dimension_scaling_relationship.ipynb` | Exploratory 100/500-tree p/m/d study: pooled accuracy association was strongest for d/p (Spearman ρ=0.593), then d/m (0.561), then absolute d (0.478); within-block signs were positive for both ratios in all eight blocks, but thresholds varied widely. |
| OpenML predictive performance | `15_openml_predictive_performance.ipynb` | 8 OpenML-CC18 datasets, up to 1,000 rows each, 5 repetitions, and 40 dataset×repetition blocks: T=1 minus RF +0.007 (95% CI [-0.003, 0.016]); T=2 -0.010 ([-0.021, 0.001]); T=3 -0.023 ([-0.037, -0.009]). |

The summary and the current rule-of-thumb section below report the latest reruns. Earlier detailed sections are retained as historical pre-rule results and should not be read as current values.

## Current rule-of-thumb rerun

The fixed-width rule was applied with an internal `RandomForestClassifier`/`RandomForestRegressor` using 100 trees, `n_jobs=-1`, and `d=20*p` computed from the actual dataset width. The fixed dimensions were p=120 → d=2400 in notebooks 01, 06, 07, 08, 10, 11, and 12; p=60 → d=1200 in notebook 05; and p=120 → d=2400 for classification versus p=80 → d=1600 for regression in notebook 09. Notebook 02 computed d dynamically as 20 times each swept feature count (320–2560 in the feature sweep).

The main current outcomes were:

| Notebook | Current result |
| --- | --- |
| 01 baseline comparison | Across 10 seeds, one-shot tree-path accuracy was 0.856 ± 0.018, full Recursive Sketch 0.844 ± 0.025, iterative without concatenation 0.829 ± 0.028, original 0.706 ± 0.054, and matched random sparse control 0.508 ± 0.024. The baseline critical-difference diagram executed successfully. |
| 02 scaling | With 100 trees and dynamic d=20p, sample-sweep wall time rose from 0.708 s at 256 samples to 1.392 s at 2048; feature-sweep wall time rose from 0.711 s at p=16 to 1.727 s at p=128. Feature-sweep peak RSS ranged from 948 to 1503 MiB. |
| 03 UMAP | The special 2D visualization remained intact; 100-tree execution generated 7 embeddings for 630 visualization rows. 1-NN UMAP accuracy ranged from 0.698 to 0.779 across representations. |
| 04 target-dimension sweep | The sweep remained intact with p=120 and 100 trees. Raw RF accuracy was 0.841; Recursive Sketch mean accuracy rose from 0.586 at d=8 to 0.845 at d=2048 and 0.847 at d=8192. The fitted asymptote was 0.938 and the smallest dimension within 0.02 of the best was 2048. |
| 05 expanding dimension | At p=60 and block d=1200, fixed accuracy was 0.860 ± 0.021, expanding 0.869 ± 0.017, and width-matched fixed 0.876 ± 0.012 (95% CI half-widths 0.0147, 0.0121, and 0.0080). The paired verdict remained inconclusive: expanding minus fixed 0.009, CI [0.001, 0.017]; versus width-matched fixed -0.007, CI [-0.021, 0.007]. |
| 06 concatenation | At p=120 and d=2400, concatenation scored 0.830 ± 0.029 versus 0.782 ± 0.042 without concatenation, a gain of 0.048. |
| 07 iteration | At p=120 and d=2400, mean accuracies for T=0/1/2/3 were 0.686/0.857/0.845/0.834; T=2 minus T=1 was -0.012 with 95% CI [-0.018, -0.007]. Total experiment time was 143.9 s. |
| 08 initial projection | At p=120 and d=2400, projected accuracy was 0.816 and 0.834 across the two seeds versus direct accuracy 0.786 and 0.834; mean loss was -0.015 and maximum loss 0.000. The fixed-rule check met the 0.05 tolerance; no CD diagram was used because only one dimension was tested. |
| 09 predictive performance | Classification p=120,d=2400: Recursive Sketch 0.830 ± 0.029, one-shot path 0.825 ± 0.016, original 0.650 ± 0.042. Regression p=80,d=1600: Recursive Sketch R²=0.439, RMSE=164.268 versus original R²=0.993, RMSE=18.208. Classification was supported; regression was not. |
| 10 tree-path value | At p=120 and d=2400, one-shot path accuracy was 0.825, Recursive Sketch 0.830, initial projection 0.651, and random sparse control 0.525. The tree-path hypothesis was supported descriptively. |
| 11 sample efficiency | At p=120 and d=2400, Recursive Sketch mean accuracy was 0.770 at 25% data versus original 0.627, and 0.830 versus 0.650 at full data. The low-data advantage was 0.143. |
| 12 cost trade-off | At p=120 and d=2400, Recursive Sketch accuracy was 0.809 versus original 0.620; Recursive Sketch wall time was 4.267 s, peak Python memory 1054.5 MiB, and serialized state 622.1 MiB. |
| 15 OpenML predictive performance | With 8 datasets, a maximum of 1,000 observations per dataset, 5 repetitions, 100-tree forests, and d=20p, mean accuracy across datasets was 0.868 for Random Forest, 0.875 for Recursive Sketch T=1, 0.858 for T=2, and 0.846 for T=3. The T=1 gain was inconclusive; T=3 was significantly below RF in the joint-block exploratory analysis. |

Critical-difference diagrams executed for the multi-configuration comparisons in notebooks 01, 04, 05, 07, 09, and 10; their ordinary mean/std or confidence-interval summaries remain the primary reporting. Notebook 08 intentionally omits a CD diagram for its single fixed dimension.

Notebook 13 was not checked as a current rerun. Its 100-tree full curve attempt was stopped after several minutes, and a resource-appropriate 40-tree retry was also stopped before clean completion because the Gaussian 8192-dimensional curve remained too expensive. No stopped 13 run contributes current numerical claims here; its prior curve outputs are historical. Notebook 14 remains the current 100/500-tree multi-factor p/m/d reference and was not collapsed to d=20p.

All successful current notebook results remain exploratory: they use controlled datasets, small seed counts in several studies, one downstream learner in most comparisons, and held-out test results during hypothesis exploration. The d=20p rule is a documented initial rule of thumb, not a validated universal selection method.

## Preliminary question — initial projection

The notebook tested whether the initial projection preserved random-forest performance within an absolute accuracy-loss tolerance of 0.05. Across two seeds and dimensions 16, 32, and 64, the largest observed loss was 0.055. The predefined criterion was therefore not met.

## Question 1 — predictive performance

Mean classification accuracy was:

| Method | Accuracy | Standard deviation |
| --- | ---: | ---: |
| Original | 0.650 | 0.042 |
| Initial random projection | 0.648 | 0.051 |
| One-shot tree-path | 0.722 | 0.085 |
| Recursive Sketch | 0.635 | 0.043 |

Mean regression results were:

| Method | R² | RMSE |
| --- | ---: | ---: |
| Original | 0.993 | 18.208 |
| Initial random projection | 0.464 | 161.986 |
| One-shot tree-path | 0.292 | 185.450 |
| Recursive Sketch | 0.117 | 205.560 |

The full Recursive Sketch method did not improve over the original representation or the initial projection in these runs, so the hypothesis was not supported.

## Question 2 — tree-path representation value

Mean classification accuracy was 0.722 for the one-shot real tree-path representation and 0.507 for the matched random sparse control. Recursive Sketch reached 0.635, while the initial random projection reached 0.648. The real path features beat the random sparse control, supporting the narrower claim that the forest paths carry useful signal.

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

Across the tested fractions, mean accuracy was 0.634 for Recursive Sketch and 0.653 for the original representation. At 25% of the training data, the Recursive Sketch advantage was -0.008. The hypothesis was not supported in this exploratory study.

## Question 6 — quality-cost trade-off

| Method | Accuracy | Wall time (s) | CPU time (s) | Peak Python memory (MB) | Serialized state (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0.620 | 1.512 | 5.108 | 1.445 | 0.005 |
| Initial random projection | 0.611 | 3.093 | 10.359 | 0.737 | 0.007 |
| Recursive Sketch | 0.605 | 5.403 | 14.997 | 30.406 | 6.305 |

Recursive Sketch was slower and used more memory while scoring below the original baseline by 0.016. The hypothesis was not supported under this cost envelope.

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

The experiments provide evidence that real tree-path features contain signal beyond a matched random sparse control. They do not yet establish that the complete iterative Recursive Sketch architecture improves predictive performance over simpler baselines. In particular, later fixed-width iterations were harmful in the current study, and expanding dimensionality did not beat a width-matched fixed representation.

## Dimension scaling relationship — 100/500-tree follow-up

The executed `14_dimension_scaling_relationship.ipynb` study used three controlled synthetic classification datasets with p=12, 48, and 120, the scikit-learn breast-cancer dataset with p=30, forests with exactly 100 and 500 trees, six target dimensions (8, 32, 128, 512, 2048, and 8192), and two paired seeds. Trees used `max_depth=8` and `n_jobs=-1`; the sweep used sparse path projection and non-materialized signed-hash concatenation. Every configuration recorded p, sample count, fitted path width m, d/p, d/m, raw-forest accuracy, Recursive Sketch accuracy, separate fit/transform/evaluation wall times, an output-memory proxy, serialized estimator size, and status. All 96 configurations completed successfully in 75.4 seconds.

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
