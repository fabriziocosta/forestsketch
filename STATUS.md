# Dimension rule-of-thumb rerun status

## What is being tried

This pass tests the initial rule of thumb that fixed-width Recursive Sketch studies should use an internal random forest with `n_estimators=100`, `n_jobs=-1`, and target dimension `d = 20 * p`, where `p` is the actual input feature count for that notebook's dataset. Each notebook must print or record its computed `p` and `d`; expected `DataDimensionalityWarning` messages may be suppressed, but real execution errors must remain visible.

Notebook 14 remains the multi-factor reference with its full p/m/d grid and is not collapsed to one fixed dimension. Notebook 03 remains a 2D UMAP visualization, and notebooks 04 and 13 remain target-dimension sweeps/curves; these special designs are not converted to one `d=20p` setting. Their forests use 100 trees where practical, while notebook 14 retains its established 100/500-tree scaling grid.

## Execution checklist

- [x] 01_compare_baselines.ipynb — fixed dataset p=120; target d=2400; 100-tree baseline comparison; executed cleanly (10 seeds).
- [x] 02_scaling_benchmarks.ipynb — target d computed as 20 times each swept feature count; 100 trees; timing/memory scaling; executed cleanly.
- [x] 03_umap_representations.ipynb — special 2D UMAP design retained; 100-tree forest; executed cleanly with 7 embeddings.
- [x] 04_q7_hypothesis_target_dimension.ipynb — dimension sweep retained; 100-tree forest; executed cleanly.
- [x] 05_q8_hypothesis_expanding_dimension.ipynb — dataset p=60; block d=1200; 100-tree forests; executed cleanly.
- [x] 06_q4_hypothesis_concatenation.ipynb — dataset p=120; d=2400; 100-tree forests; executed cleanly.
- [x] 07_q3_hypothesis_iteration.ipynb — dataset p=120; d=2400; 100-tree forests; executed cleanly.
- [x] 08_preliminary_hypothesis_initial_projection.ipynb — fixed rule-of-thumb d=2400; 100-tree forests; executed successfully, maximum observed loss 0.000 and no CD diagram (one fixed dimension).
- [x] 09_q1_hypothesis_predictive_performance.ipynb — classification p=120/d=2400; regression p=80/d=1600; 100-tree forests; executed cleanly.
- [x] 10_q2_hypothesis_tree_path_value.ipynb — dataset p=120; d=2400; 100-tree forests; executed cleanly.
- [x] 11_q5_hypothesis_sample_efficiency.ipynb — dataset p=120; d=2400; 100-tree forests; executed cleanly.
- [x] 12_q6_hypothesis_cost_tradeoff.ipynb — dataset p=120; d=2400; 100-tree forests; executed cleanly.
- [ ] 13_dimension_iteration_curves.ipynb — special curves retained; 100-tree attempt and 40-tree retry both stopped for resource/time cost before clean completion; no current result claimed.
- [x] 14_dimension_scaling_relationship.ipynb — special multi-factor p/m/d reference retained at 100/500 trees; existing full run verified cleanly.
- [x] Core pytest suite — 11 tests passed under Python 3.12.13 in 1.15 seconds.

## Notes

Items will be checked only after the corresponding notebook has executed without errors and its actual results have been incorporated into Results.md. Existing dirty notebook edits from prior work are preserved and are not part of this checklist unless explicitly changed for this pass.
