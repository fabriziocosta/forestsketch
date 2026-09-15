# TODO

Prioritized follow-up work for turning the current Forest Sketch research prototype into a predictable, extensible scikit-learn component.

## P0 — API correctness and contracts

- [x] Define and test the output contract for `dimension_mode="fixed"` and `dimension_mode="expanding"`.
  - Fixed output width: `n_components`.
  - Expanding output width: `n_components * (n_iterations + 1)`.
  - [x] Expose the fitted width as `output_dimension_`.
- [x] Make normalization stage-specific with explicit `input_normalizer`, `path_normalizer`, and `concat_normalizer` components.
- [x] Apply the configured concatenation normalizer before the next forest in expanding mode.
- [x] Add `output_format="dense" | "sparse" | "auto"` and test the return type for every projector and dimension mode.
- [x] Override `fit_transform` so the representation computed during `fit` is reused instead of transforming the training data a second time.
- [x] Add `sample_weight=None` to `fit` and forward it to every cloned random forest.
- [x] Use scikit-learn validation helpers for `X`, `y`, feature names, and sample counts, with clear errors for malformed inputs.

## P0 — Test coverage

- [ ] Test fixed and expanding modes in a `Pipeline` with both classifiers and regressors.
- [ ] Test sparse input, sparse-safe normalization, and signed-hash output in both modes.
- [ ] Test deterministic repeated fitting and transformation with fixed forest and projection seeds.
- [ ] Test `clone`, `get_params`, `set_params`, `get_feature_names_out`, and pre-fit errors.
- [ ] Test invalid dimensions, iteration counts, projection types, normalization settings, and dimension modes.
- [ ] Test custom path encoders and normalizers through the public extension interface.
- [ ] Add a regression test proving that the caller-supplied forest is not fitted or mutated.

## P1 — Extensibility and metadata

- [ ] Allow cloneable projector templates or factories for the initial, path, and concatenation stages.
- [ ] Preserve the current string-based projector shortcuts as convenient defaults.
- [ ] Store `feature_names_in_`, `output_dimension_`, per-stage widths, and projection seeds after fitting.
- [ ] Generate block-aware feature names such as `forestsketch_block1_0` in expanding mode.

## P1 — Scalability and reliability

- [ ] Add batching for path extraction and projection so large sparse forests do not require all intermediate matrices at once.
- [ ] Measure and document memory growth for materialized Gaussian, sparse, and signed-hash projectors.
- [ ] Add resource-aware tests for high target dimensions and expanding representations.
- [ ] Verify serialization and deserialization across supported Python and scikit-learn versions.

## P2 — Scientific validation

- [ ] Re-run the fixed-versus-expanding study with cross-validation and a width-matched control.
- [ ] Use paired seeds and confidence intervals for all iteration and dimension sweeps.
- [x] Report the raw forest accuracy and the uncompressed visited-node width beside every relevant compressed result.
- [ ] Separate wall-clock time, CPU time, peak memory, output width, and serialized model size in experiment reports.
- [ ] Keep the final test set untouched while selecting `n_components`, `n_iterations`, and dimension mode.
- [x] Run a paired multi-dataset study of target dimension `d` versus input width `p` and fitted path width `m` before changing the other hypothesis notebooks.
- [ ] Add experiments across multiple datasets and task types before making performance claims.

## P2 — Packaging and documentation

- [ ] Add API reference documentation with examples for fixed and expanding modes.
- [x] Document that the supplied forest controls `n_jobs`; experiment helpers use `n_jobs=-1` for full parallelism.
- [ ] Add a release checklist covering tests, notebook execution, changelog, semantic versioning, and GitHub synchronization.

## Completed project organization

- [x] Reorder notebooks so the UMAP study is notebook 03 and the related dimensionality, expansion, concatenation, and iteration studies are adjacent.
- [x] Add `Results.md` with the executed result for every hypothesis and the preliminary initial-projection study.
- [x] Add an executed dimensionality-scaling study varying input width, forest size, and projected width.
