from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "scania_aps_failure.ipynb"
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # APS Failure at Scania Trucks

        This notebook repurposes the structure of the Credit Risk Classification assignment for a real operational-maintenance problem.

        The positive class is `pos`: a failure of the APS component in focus. The negative class is `neg`: a truck with a failure unrelated to that APS component. The model ranks trucks for inspection; it does not automatically declare a truck safe or unsafe.

        **Official source:** [UCI APS Failure at Scania Trucks](https://archive.ics.uci.edu/dataset/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks), DOI [10.24432/C51S51](https://doi.org/10.24432/C51S51).

        **Official challenge costs:** 10 cost units for an unnecessary mechanic check and 500 cost units for missing a faulty truck. These are challenge units, not euros and not measured realised savings.
        """
    ),
    md(
        """
        ## 1. Import data and libraries

        The notebook can run from the project folder. If the official CSV is not present, it downloads the official UCI archive into the relative `data/raw` directory. No personal path is required.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import platform
        import sys
        import urllib.request
        import zipfile

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_curve

        RANDOM_STATE = 77
        SOURCE_URL = 'https://archive.ics.uci.edu/static/public/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks.zip'

        def find_project_root(start=Path.cwd()):
            for candidate in (start, *start.parents):
                if (candidate / 'src' / 'data_contract.py').exists() and (candidate / 'data' / 'raw').exists():
                    return candidate
            raise FileNotFoundError('Run this notebook from the scania_aps_classification project folder.')

        ROOT = find_project_root()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        RAW_DIR = ROOT / 'data' / 'raw'
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        TRAIN_PATH = RAW_DIR / 'aps_failure_training_set.csv'
        OFFICIAL_TEST_PATH = RAW_DIR / 'aps_failure_test_set.csv'
        ZIP_PATH = RAW_DIR / 'aps_failure_at_scania_trucks.zip'

        if not TRAIN_PATH.exists() or not OFFICIAL_TEST_PATH.exists():
            request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=180) as response, ZIP_PATH.open('wb') as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if ZIP_PATH.stat().st_size < 1_000_000:
                raise RuntimeError('The UCI download is unexpectedly small.')
            with zipfile.ZipFile(ZIP_PATH) as archive:
                archive.extractall(RAW_DIR)

        from src.data_contract import contract_summary, load_raw_data, make_xy, validate_schema
        from src.cost_policy import (
            COST_MISSED_FAILURE,
            COST_UNNECESSARY_INSPECTION,
            challenge_metrics,
            cost_sweep,
            top_k_metrics,
        )
        from src.evaluation import binary_metrics, evaluate_model_set
        from src.models import make_candidate_models
        from src.preprocessing import build_outlier_audit

        print(f'Project root: {ROOT}')
        print(f'Training source: {TRAIN_PATH}')
        print(f'Official frozen-test source: {OFFICIAL_TEST_PATH}')
        print(f'Python: {platform.python_version()}')
        """
    ),
    md("""
        ## 2. Load data, define the target and split before fitting anything

        The UCI files contain a 20-line licence preamble, which is skipped by the data contract. Missing values are encoded as `na` and become `NaN`.

        The UCI archive supplies separate labelled training and test files. The full labelled training file is used for development out-of-fold model and threshold selection. The separate labelled official test file is kept untouched and used once for final confirmation.
        """),
    code(
        """
        training_frame = load_raw_data(TRAIN_PATH)
        official_test_frame = load_raw_data(OFFICIAL_TEST_PATH)
        print('Training-data preview:')
        display(training_frame.head())
        print(f'Training shape: {training_frame.shape}; official test shape: {official_test_frame.shape}')
        validate_schema(training_frame, require_full_training_size=True)
        validate_schema(official_test_frame)
        X_dev, y_dev = make_xy(training_frame)
        X_test, y_test = make_xy(official_test_frame)
        X_dev = X_dev.astype('float32')
        X_test = X_test.astype('float32')

        split_report = pd.DataFrame({
            'split': ['labelled development/training', 'official frozen test'],
            'rows': [len(y_dev), len(y_test)],
            'positive APS failures': [int(y_dev.sum()), int(y_test.sum())],
            'positive rate': [y_dev.mean(), y_test.mean()],
        })
        display(split_report)
        print('The positive class is APS component failure; the negative class is an unrelated component failure.')
        """
    ),
    md("""
        ## 3. Data contract and context

        One row represents one selected truck operating example. The anonymised variables are operational counters and histogram bins. The public data does not provide a future time horizon, maintenance outcome, or causal proof that an inspection prevented a breakdown. Therefore, the target is a benchmark classification label, not a directly observed “failure in the next 12 months” outcome.
        """),
    code(
        """
        training_summary = contract_summary(training_frame)
        test_summary = contract_summary(official_test_frame)
        contract_table = pd.DataFrame([
            {'dataset': 'labelled development/training', 'check': check, 'result': value}
            for check, value in training_summary.items()
        ] + [
            {'dataset': 'official frozen test', 'check': check, 'result': value}
            for check, value in test_summary.items()
        ])
        display(contract_table)
        assert training_summary['rows'] == 60_000
        assert training_summary['features'] == 170
        assert training_summary['positive_failures'] == 1_000
        assert test_summary['rows'] == 16_000
        assert test_summary['positive_failures'] == 375
        """
    ),
    md("""
        ## 4. Data cleaning and missingness audit

        Missing sensor values are part of the operational-data problem. We do not delete rows or fill values using the whole dataset. Median imputation and missingness indicators are fitted inside each model pipeline using development data only. Duplicate rows and constant predictors are checked before modelling.

        Outliers are detected feature by feature on development data. Approximately symmetric features use a z-score bound of +/-3 standard deviations. Skewed/non-normal features use Tukey's IQR rule (1.5 x IQR). We cap extreme values inside each training fold rather than deleting trucks; the official test remains untouched until confirmation.
        """),
    code(
        """
        cleaning_report = pd.DataFrame({
            'check': ['missing cells', 'duplicate rows', 'constant predictors', 'positive label count'],
            'count': [
                int(X_dev.isna().sum().sum()),
                int(training_frame.duplicated().sum()),
                int((X_dev.nunique(dropna=False) <= 1).sum()),
                int(y_dev.sum()),
            ],
        })
        display(cleaning_report)

        FIGURES = ROOT / 'figures'
        TABLES = ROOT / 'artifacts' / 'tables'
        FIGURES.mkdir(parents=True, exist_ok=True)
        TABLES.mkdir(parents=True, exist_ok=True)

        missingness = X_dev.isna().mean().sort_values(ascending=False)
        display(missingness.head(15).rename('missing share').to_frame())
        plt.figure(figsize=(10, 4))
        missingness.head(25).sort_values().plot.barh(color='#4c78a8')
        plt.title('Development missingness: 25 most incomplete operational features')
        plt.xlabel('Share of rows missing')
        plt.tight_layout()
        plt.savefig(FIGURES / 'missingness.png', dpi=160)
        plt.show()

        outlier_audit = build_outlier_audit(X_dev)
        display(outlier_audit.head(15))
        display(outlier_audit.groupby('method', dropna=False).agg(
            features=('feature', 'count'),
            flagged_values=('flagged_count', 'sum'),
            average_flagged_rate=('flagged_rate', 'mean'),
        ).reset_index())
        outlier_audit.to_csv(TABLES / 'outlier-audit.csv', index=False)
        plt.figure(figsize=(10, 4))
        outlier_audit.head(20).sort_values('flagged_rate').plot.barh(
            x='feature', y='flagged_rate', legend=False, color='#f58518'
        )
        plt.title('Development outlier flags: 20 highest feature rates')
        plt.xlabel('Share of observed values flagged')
        plt.tight_layout()
        plt.savefig(FIGURES / 'outlier-flags.png', dpi=160)
        plt.show()
        del training_frame, official_test_frame
        """
    ),
    md("""
        ## 5. Candidate models and leakage-safe pipelines

        The same comparison logic is retained from the credit-risk assignment: a majority baseline, logistic regression, a decision tree and a random forest. The rare positive class requires class weighting for the non-baseline candidates. Outlier clipping, imputation, missingness indicators and scaling are inside the pipeline, so cross-validation cannot learn from the validation fold.
        """),
    code(
        """
        models = make_candidate_models(list(X_dev.columns), random_state=RANDOM_STATE)
        print('Candidate models:', list(models))
        """
    ),
    md("""
        ## 6. Fit, compare and select using development out-of-fold predictions

        ROC-AUC and average precision measure ranking quality. The Scania decision criterion is total challenge cost: false positives cost 10, while false negatives cost 500. The official frozen test is not used to select the model.
        """),
    code(
        """
        comparison, predictions, fitted_models = evaluate_model_set(
            models,
            X_dev,
            y_dev,
            X_test,
            y_test,
            random_state=RANDOM_STATE,
        )

        development_rows = []
        for model_name in models:
            score = predictions[model_name]['development']
            sweep = cost_sweep(y_dev, score, np.linspace(0.001, 0.999, 200))
            best = sweep.sort_values(['total_cost', 'fn', 'fp']).iloc[0]
            metric_row = comparison.query("model == @model_name and split == 'development OOF'").iloc[0]
            development_rows.append({
                'model': model_name,
                'ROC-AUC': metric_row['roc_auc'],
                'average precision': metric_row['average_precision'],
                'best development threshold': best['threshold'],
                'development total cost': best['total_cost'],
                'development cost avoided': best['cost_avoided_vs_all_negative'],
                'inspections at best threshold': best['inspections'],
            })
        model_selection = pd.DataFrame(development_rows).sort_values(
            ['development total cost', 'average precision'], ascending=[True, False]
        ).reset_index(drop=True)
        display(model_selection)
        selected_model = model_selection.iloc[0]['model']
        selected_dev_score = predictions[selected_model]['development']
        selected_test_score = predictions[selected_model]['frozen_test']
        print(f'Selected model from development cost: {selected_model}')
        model_selection.to_csv(TABLES / 'model-selection.csv', index=False)
        """
    ),
    md("""
        ## 7. Frozen-test discrimination: ROC-AUC and precision-recall

        These curves show the model’s ranking behaviour over many possible cutoffs. They are not one selected threshold. Average precision should be compared with the positive-class base rate, because the class is highly imbalanced.
        """),
    code(
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        test_metrics = binary_metrics(y_test, selected_test_score, threshold=0.5)
        print(f"Frozen-test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Frozen-test average precision: {test_metrics['average_precision']:.4f}")
        print(f"Frozen-test positive base rate: {y_test.mean():.4f}")

        fpr, tpr, _ = roc_curve(y_test, selected_test_score)
        precision, recall, _ = precision_recall_curve(y_test, selected_test_score)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], '--', color='grey', label='random benchmark')
        axes[0].set(xlabel='False-positive rate', ylabel='Recall / true-positive rate', title=f'{selected_model}: frozen-test ROC curve')
        axes[0].legend()
        axes[1].plot(recall, precision, label=f"Average precision = {test_metrics['average_precision']:.3f}")
        axes[1].axhline(y_test.mean(), linestyle='--', color='grey', label='positive base rate')
        axes[1].set(xlabel='Recall', ylabel='Precision', title=f'{selected_model}: frozen-test precision-recall curve')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(FIGURES / 'roc-pr-curves.png', dpi=160)
        plt.show()
        """
    ),
    md("""
        ## 8. Six operating points and the required cost table

        A threshold is the score cutoff for sending a truck to inspection. Lowering it usually captures more positive failures but sends more unrelated-failure trucks to a mechanic. Because the UCI case does not specify a real inspection capacity, the 10% point below is explicitly an illustrative capacity scenario, not a Scania fact.
        """),
    code(
        """
        dev_sweep = cost_sweep(y_dev, selected_dev_score, np.linspace(0.001, 0.999, 400))
        cost_optimal_threshold = float(dev_sweep.sort_values(['total_cost', 'fn', 'fp']).iloc[0]['threshold'])

        def threshold_for_recall(y_true, score, target=0.90):
            candidates = np.unique(np.round(score, 6))
            rows = [challenge_metrics(y_true, score, float(t)) for t in candidates]
            eligible = [row for row in rows if row['recall'] >= target and row['inspections'] > 0]
            return float(max(eligible, key=lambda row: row['threshold'])['threshold']) if eligible else float(np.quantile(score, 0.20))

        def threshold_for_precision(y_true, score, target=0.80, min_inspections=25):
            candidates = np.unique(np.round(score, 6))
            rows = [challenge_metrics(y_true, score, float(t)) for t in candidates]
            eligible = [row for row in rows if row['precision'] >= target and row['inspections'] >= min_inspections]
            return float(min(eligible, key=lambda row: row['threshold'])['threshold']) if eligible else float(np.quantile(score, 0.95))

        high_recall_threshold = threshold_for_recall(y_dev, selected_dev_score)
        high_precision_threshold = threshold_for_precision(y_dev, selected_dev_score)
        mid_threshold = 0.20
        capacity_fraction = 0.10
        capacity_threshold = float(np.quantile(selected_dev_score, 1 - capacity_fraction))

        points = [
            {'name': 'default threshold 0.50', 'threshold': 0.50, 'reason': 'Required reference point where meaningful'},
            {'name': 'high-recall point', 'threshold': high_recall_threshold, 'reason': 'Development threshold targeting at least 90% recall'},
            {'name': 'mid-range point', 'threshold': mid_threshold, 'reason': 'Additional operating point'},
            {'name': 'high-precision point', 'threshold': high_precision_threshold, 'reason': 'Development threshold targeting at least 80% precision where feasible'},
            {'name': 'cost-aware point', 'threshold': cost_optimal_threshold, 'reason': 'Lowest development challenge cost'},
            {'name': 'capacity-constrained ranking point', 'threshold': capacity_threshold, 'reason': 'Illustrative top-10% inspection capacity'},
        ]

        table_rows = []
        for point in points:
            row = challenge_metrics(y_test, selected_test_score, point['threshold'])
            row.update({'operating point': point['name'], 'threshold rule': f"score >= {point['threshold']:.6f}", 'reason': point['reason']})
            table_rows.append(row)
        capacity_test = top_k_metrics(y_test, selected_test_score, capacity_fraction)
        capacity_test.update({'operating point': 'capacity-constrained exact top-10%', 'threshold rule': 'rank highest 10% exactly', 'reason': 'Exact capacity policy; boundary score is diagnostic'})
        threshold_table = pd.DataFrame(table_rows + [capacity_test])
        threshold_table = threshold_table[[
            'operating point', 'threshold rule', 'reason', 'inspections', 'inspection_rate', 'tp', 'fp', 'fn', 'tn',
            'precision', 'recall', 'f1', 'unnecessary_inspection_cost', 'missed_failure_cost', 'total_cost',
            'all_negative_baseline_cost', 'cost_avoided_vs_all_negative'
        ]]
        display(threshold_table)
        threshold_table.to_csv(TABLES / 'frozen-test-threshold-table.csv', index=False)
        """
    ),
    md("""
        **How to read the table:** `total cost = 10 × false positives + 500 × false negatives`. The recommended policy is selected from development evidence, then reported once on the official frozen test. The exact top-10% row is a capacity policy; its displayed boundary score is not itself the deployment rule.
        """),
    code(
        """
        recommended_name = 'cost-aware point'
        recommended_row = threshold_table.loc[threshold_table['operating point'].eq(recommended_name)].iloc[0]
        print(f"Recommended policy: {selected_model}, score >= {recommended_row['threshold rule'].split('>=')[-1].strip()}.")
        print(f"Frozen-test total challenge cost: {recommended_row['total_cost']:.0f} units.")
        print(f"Frozen-test cost avoided versus inspecting nobody: {recommended_row['cost_avoided_vs_all_negative']:.0f} units.")
        print('This is conditional benchmark arithmetic, not measured savings or proof of causal breakdown prevention.')

        plt.figure(figsize=(9, 5))
        plt.plot(dev_sweep['threshold'], dev_sweep['total_cost'], color='#d62728')
        plt.axvline(cost_optimal_threshold, linestyle='--', color='black', label=f'development optimum = {cost_optimal_threshold:.3f}')
        plt.xlabel('Score threshold')
        plt.ylabel('Total challenge cost units')
        plt.title('Development cost across operating thresholds')
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / 'cost-by-threshold.png', dpi=160)
        plt.show()

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            (selected_test_score >= float(recommended_row['threshold rule'].split('>=')[-1])).astype(int),
            display_labels=['unrelated failure', 'APS failure'],
            cmap='Blues',
        )
        plt.title('Frozen-test confusion matrix at recommended policy')
        plt.tight_layout()
        plt.savefig(FIGURES / 'recommended-confusion-matrix.png', dpi=160)
        plt.show()
        """
    ),
    md("""
        ## 9. Sensitivity analysis

        Sensitivity changes the economic assumptions, not the model’s predictions. The base case uses the official challenge costs. The other cases show whether the recommendation is still attractive if the cost of a missed failure or an unnecessary inspection changes. All values remain challenge cost units.
        """),
    code(
        """
        recommended_threshold = float(recommended_row['threshold rule'].split('>=')[-1])
        scenarios = [
            {'scenario': 'official base', 'cost_fp': 10, 'cost_fn': 500},
            {'scenario': 'lower missed-failure cost', 'cost_fp': 10, 'cost_fn': 350},
            {'scenario': 'severe missed-failure cost', 'cost_fp': 10, 'cost_fn': 750},
            {'scenario': 'expensive inspection', 'cost_fp': 50, 'cost_fn': 500},
        ]
        sensitivity_rows = []
        for scenario in scenarios:
            row = challenge_metrics(y_test, selected_test_score, recommended_threshold, scenario['cost_fp'], scenario['cost_fn'])
            row.update(scenario)
            sensitivity_rows.append(row)
        sensitivity_table = pd.DataFrame(sensitivity_rows)[[
            'scenario', 'cost_fp', 'cost_fn', 'inspections', 'tp', 'fp', 'fn', 'precision', 'recall', 'total_cost', 'cost_avoided_vs_all_negative'
        ]]
        display(sensitivity_table)
        sensitivity_table.to_csv(TABLES / 'sensitivity-analysis.csv', index=False)

        capacity_rows = []
        for fraction in [0.05, 0.10, 0.20, 0.30]:
            row = top_k_metrics(y_test, selected_test_score, fraction)
            row['capacity'] = f'top {fraction:.0%}'
            capacity_rows.append(row)
        capacity_table = pd.DataFrame(capacity_rows)
        display(capacity_table[['capacity', 'inspections', 'tp', 'fp', 'fn', 'precision', 'recall', 'total_cost', 'cost_avoided_vs_all_negative']])
        capacity_table.to_csv(TABLES / 'capacity-sensitivity.csv', index=False)
        """
    ),
    md("""
        ## 10. Governance and operational subgroup audit

        The predictors are anonymised operational variables; the public dataset does not provide protected-group attributes. Therefore this is not a fairness certification. As an audit-only diagnostic, we compare trucks with higher versus lower missingness burden. The grouping is defined from development data and is not used for model or threshold selection.
        """),
    code(
        """
        dev_missing_cut = X_dev.isna().mean(axis=1).quantile(0.75)
        test_missing_share = X_test.isna().mean(axis=1)
        audit_group = np.where(test_missing_share >= dev_missing_cut, 'higher missingness burden', 'lower missingness burden')
        audit_rows = []
        for group in ['lower missingness burden', 'higher missingness burden']:
            mask = audit_group == group
            metrics = challenge_metrics(y_test[mask], selected_test_score[mask], recommended_threshold)
            metrics.update({'audit_group': group, 'rows': int(mask.sum()), 'positive_rate': float(y_test[mask].mean())})
            audit_rows.append(metrics)
        subgroup_audit = pd.DataFrame(audit_rows)
        display(subgroup_audit[['audit_group', 'rows', 'positive_rate', 'inspections', 'inspection_rate', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall']])
        subgroup_audit.to_csv(TABLES / 'operational-subgroup-audit.csv', index=False)
        print('Audit-only diagnostic: no protected attributes are available, and this table is not a fairness certification.')
        """
    ),
    md("""
        ## 11. Reproducibility and limitations

        The run uses random seed 77, development out-of-fold predictions on the full labelled training file for model/policy selection, and one confirmation on the separate labelled official test file. Outlier bounds, imputation values and scaling parameters are fitted inside each development fold. The official test file is never used to fit preprocessing, select the model or tune the threshold.

        The target is a benchmark class label, not a future failure horizon. Before real deployment, validate timestamp availability, prospective failure outcomes, inspection effectiveness, operational drift, alarm fatigue and whether the model changes maintenance decisions safely. Human maintenance oversight remains required.
        """),
    code(
        """
        comparison.to_csv(TABLES / 'model-metrics-development-and-frozen.csv', index=False)
        split_report.to_csv(TABLES / 'split-report.csv', index=False)
        contract_table.to_csv(TABLES / 'data-contract-summary.csv', index=False)

        manifest = {
            'dataset': 'UCI APS Failure at Scania Trucks',
            'source_url': 'https://archive.ics.uci.edu/dataset/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks',
            'doi': '10.24432/C51S51',
            'random_state': RANDOM_STATE,
            'split': '60,000 labelled training rows for development; 16,000 labelled official test rows for frozen confirmation',
            'selected_model': selected_model,
            'recommended_policy': 'development cost-aware threshold, confirmed once on official frozen test',
            'outlier_policy': 'feature-wise z-score +/-3 for approximately symmetric features; Tukey IQR 1.5x for skewed/non-normal features; clip inside each training fold',
            'cost_unnecessary_inspection': COST_UNNECESSARY_INSPECTION,
            'cost_missed_failure': COST_MISSED_FAILURE,
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'sklearn': __import__('sklearn').__version__,
        }
        (ROOT / 'artifacts').mkdir(exist_ok=True)
        (ROOT / 'artifacts' / 'run-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(manifest, indent=2))
        """
    ),
    md("""
        ## Conclusion

        The final recommendation is a cost-sensitive inspection-prioritisation policy. It ranks trucks by APS-failure score and uses a threshold selected on development data to reduce the much larger cost of missed failures. The result is a benchmark decision aid, not an automatic maintenance decision and not proof of realised savings.
        """),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    },
)
nbf.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
