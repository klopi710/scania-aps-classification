# Reproducibility note

- Source: UCI Machine Learning Repository dataset 421, APS Failure at Scania Trucks.
- Data download: official UCI static archive, downloaded by the notebook when missing.
- Random seed: 77.
- Label mapping: `pos` → 1 (APS component failure), `neg` → 0 (unrelated component failure).
- Development/frozen split: the full 60,000-row labelled training file supplies development out-of-fold predictions; the separate 16,000-row labelled UCI test file is the frozen confirmation set and is not used for model or policy selection.
- Missing values: `na` treated as missing and imputed inside pipelines fitted only on development folds.
- Costs: Cost_1 = 10 for unnecessary inspection; Cost_2 = 500 for missed faulty truck.
- Execution: run the notebook from `scania_aps_classification/` with a clean kernel, or execute the command in `README.md`.

The official UCI test file is scored once after the model and policy are frozen. Its labels are not used to fit preprocessing, select the model or tune the threshold.
