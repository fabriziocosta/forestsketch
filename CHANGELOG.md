# Changelog

All notable changes to Recursive Sketch are documented here.

## 1.3.0

- Added explicit estimator output formatting with dense, sparse, and automatic modes.
- Added sample-weight forwarding and an efficient `fit_transform` implementation.
- Added fitted output-dimension metadata and applied normalization consistently in expanding mode.
- Simplified the public API to return only final representations; intermediate matrices remain internal.
- Added the prioritized `TODO.md` and compiled experimental findings in `Results.md`.
- Reordered and renamed experiment notebooks by hypothesis question, with fully parallel forest helpers.

## 1.0.0

- First automated public GitHub release.
- Includes the initial `RecursiveSketchClassifier` implementation and experiment notebook.

## 0.1.0

- Initial public implementation of `RecursiveSketchClassifier`.
- Modular path encoding, normalization, and projection components.
- Optional intermediate-representation inspection.
