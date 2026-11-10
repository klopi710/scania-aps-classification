# Scania APS Failure Classification

This project applies the workflow used in the credit-risk assignment to the UCI APS Failure at Scania Trucks dataset.

The positive class is a truck with a failure of the APS component in focus. The negative class is a truck with a failure unrelated to that APS component. The model is an inspection-prioritisation aid: it ranks trucks by estimated APS-failure risk so maintenance teams can decide which cases deserve attention first. It does not automatically declare a truck safe or unsafe.

The official challenge cost matrix is used instead of the credit-risk business assumptions:

- Cost 10: unnecessary mechanic inspection.
- Cost 500: missed faulty truck, which may lead to a breakdown.

These are challenge cost units, not euros and not measured realised savings. The UCI page states that the labelled training file contains 60,000 examples and 171 total attributes; the downloaded official test file contains 16,000 labelled examples. The training file has 59,000 negative examples and 1,000 positive examples; the official test file has 15,625 negative and 375 positive examples. Missing values are represented by `na`.

The official data source is the [UCI APS Failure at Scania Trucks dataset](https://archive.ics.uci.edu/dataset/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks), DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51). It is licensed under CC BY 4.0.

## Run

From this folder:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m nbconvert --to notebook --execute notebooks/scania_aps_failure.ipynb --output scania_aps_failure.executed.ipynb --ExecutePreprocessor.timeout=900
python tools/build_scania_business_deck.py
```

The notebook displays the raw `head()` preview and then a reader-friendly renamed preview. The feature names remain anonymised because the public UCI release does not disclose the physical meaning of each individual signal. The renamed preview and its explanation are also saved as `data/processed/renamed_dataset_preview.csv` and `data/processed/preview_column_guide.csv`. In each row, `class=pos` means failure of the APS component in focus, `class=neg` means an unrelated component failure, and `NaN` means a missing measurement. The notebook then downloads the official UCI archive into `data/raw/` if either labelled CSV is absent. The full labelled training file is used for development out-of-fold model and threshold selection; the separate labelled official test file is used once for frozen confirmation. Feature-wise outliers are capped inside each training fold: approximately symmetric features use z-score +/-3, while skewed/non-normal features use Tukey's 1.5 x IQR rule. Rows are not deleted. Running it from a clean kernel creates the figures, tables and manifest under this project folder.

## Important limitation

The UCI labels identify the challenge classes, not a future time-stamped failure horizon. The analysis therefore demonstrates a defensible benchmark workflow and cost-sensitive ranking, not proof of real-world causal breakdown prevention. Deployment would require timestamped prospective validation, inspection outcomes, drift monitoring and confirmation that the available features exist before the inspection decision.

The presentation deck is generated at `slides/image/scania-aps-business-deck/scania-aps-business-deck.pptx`. It contains 10 main slides and 4 technical appendix slides, using the dark card-based format from the individual credit-risk presentation.
