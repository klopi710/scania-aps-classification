# APS Failure at Scania Trucks

## Slide 1 — Decision

Use a Random Forest to rank trucks by estimated APS-failure risk and prioritise mechanic inspection.

The development-selected cost-aware threshold was approximately **0.096**. On the official frozen test, it inspected **1,037 of 16,000 trucks**, captured **367 of 375 APS-failure cases**, and missed **8**.

This is a maintenance prioritisation aid. It does not automatically declare a truck safe, unsafe or ready to operate. A mechanic remains responsible for the final inspection decision.

## Slide 2 — Why this matters

The Air Pressure System supports functions such as braking and gear changes.

- The positive class is an APS component failure.
- The negative class is a truck with a failure unrelated to that APS component.
- Missing a faulty truck may cause a breakdown.
- An unnecessary mechanic check consumes time and capacity.

The costs are deliberately asymmetric:

- Unnecessary inspection: **10 challenge cost units**.
- Missed faulty truck: **500 challenge cost units**.

The missed failure is therefore **50 times more expensive** than the unnecessary inspection.

## Slide 3 — Data and modelling approach

The UCI training file contains:

- **60,000 labelled truck examples**;
- **170 anonymised operational predictors**;
- **1,000 positive APS failures** and **59,000 negative examples**;
- substantial missing sensor values, represented by `na`.

We used the full 60,000-row labelled training file for development out-of-fold selection and the separate labelled 16,000-row official UCI test file for one frozen confirmation. Missing values and missingness indicators are fitted inside leakage-safe pipelines using development data only.

The comparison includes a majority baseline, Logistic Regression, Decision Tree and Random Forest.

## Slide 4 — Model selection

Random Forest was selected using development out-of-fold evidence and the Scania cost criterion.

| Model | Development ROC-AUC | Development average precision | Best development cost | Cost avoided vs all-negative baseline |
|---|---:|---:|---:|---:|
| Random Forest | 0.981 | 0.736 | 44,080 | 455,920 |
| Logistic Regression | 0.943 | 0.713 | 79,870 | 420,130 |
| Decision Tree | 0.892 | 0.463 | 79,360 | 420,640 |
| Majority baseline | 0.500 | 0.017 | 500,000 | 0 |

The official test was not used to choose the model or threshold.

## Slide 5 — Official frozen-test result

At the development-selected Random Forest threshold of approximately **0.096**:

- **Inspected:** 1,037 of 16,000 trucks;
- **True positives:** 367;
- **False positives:** 670;
- **False negatives:** 8;
- **True negatives:** 14,955;
- **Precision:** 35.4%;
- **Recall:** 97.9%;
- **Total cost:** 10,700 challenge units;
- **Cost avoided versus inspecting nobody:** 176,800 challenge units.

This is an official frozen-test confirmation, not measured real-world savings.

## Slide 6 — Why the threshold is not automatically 0.50

The threshold controls the trade-off:

- At **0.50**, the official test inspects 467 trucks, captures 316 failures and misses 59. Total cost: **31,010**.
- At **0.096**, the official test inspects 1,037 trucks, captures 367 failures and misses 8. Total cost: **10,700**.
- At an illustrative **top-10% capacity policy**, the official test inspects exactly 1,600 trucks, captures 373 failures and misses 2. Total cost: **13,270**.

The selected threshold is lower because a missed APS failure is much more expensive than an unnecessary inspection.

## Slide 7 — Recommendation and safeguards

Use Random Forest as an inspection-prioritisation layer alongside the existing maintenance workflow.

1. Rank trucks by APS-failure score.
2. Send cases above the validated operating threshold to mechanic inspection.
3. Keep human decision authority and record inspection outcomes.
4. Monitor false negatives, inspection workload, missingness, calibration and drift.
5. Revalidate prospectively before changing maintenance scheduling or declaring the model production-ready.

The UCI label is a benchmark class, not a future time-stamped failure outcome. The cost result is conditional challenge-unit arithmetic.

## Slide 8 — Final answer

The public Scania dataset supports a meaningful classification case because it combines many operational predictors, a real positive failure class, severe imbalance and an explicit asymmetric cost function.

Random Forest gives the strongest development ranking and cost result in this implementation. The model appears useful for prioritising scarce mechanic attention, but one official test set is not enough to prove deployment performance.

**Operational next step:** obtain timestamped prospective outcomes and test whether the inspection policy reduces missed APS failures without creating unacceptable mechanic workload.

## Appendix 1 — Metrics and curves

ROC-AUC evaluates how well the model orders positive APS failures above negative examples across possible score cutoffs. Average precision summarises the precision-recall trade-off and should be compared with the positive base rate of approximately 1.67%.

The ROC and precision-recall curves are generated in `figures/roc-pr-curves.png`.

## Appendix 2 — Required operating-point table

The executed notebook contains the full table with threshold rule, inspection count/rate, TP, FP, FN, TN, precision, recall, F1, unnecessary-inspection cost, missed-failure cost, total cost and cost avoided.

The six diagnostic points are: threshold 0.50, high recall, mid-range, high precision, cost-aware, and illustrative capacity-constrained ranking. The capacity point is labelled illustrative because the public UCI case does not specify an actual inspection capacity.

## Appendix 3 — Sensitivity analysis

At the recommended official frozen-test policy:

| Scenario | Unnecessary inspection cost | Missed failure cost | Total cost | Cost avoided vs all-negative |
|---|---:|---:|---:|---:|
| Official base | 10 | 500 | 10,700 | 176,800 |
| Lower missed-failure cost | 10 | 350 | 9,500 | 121,750 |
| Severe missed-failure cost | 10 | 750 | 12,700 | 268,550 |
| Expensive inspection | 50 | 500 | 37,500 | 150,000 |

The recommendation remains cost-saving under these conditional scenarios, but the arithmetic does not establish causal savings.

## Appendix 4 — Governance and reproducibility

- The predictors are anonymised operational variables; protected-group attributes are not supplied.
- The subgroup audit therefore compares higher versus lower missingness burden only. It is not a fairness certification.
- Seed: **77**.
- Split: 60,000 labelled training rows for development; 16,000 labelled official test rows for frozen confirmation.
- Missing values: median imputation and missingness indicators inside pipelines.
- The official 16,000-row UCI test file is labelled and is scored once after model and threshold selection.
- Notebook: `notebooks/scania_aps_failure.ipynb`.
- Executed notebook: `notebooks/scania_aps_failure.executed.ipynb`.
