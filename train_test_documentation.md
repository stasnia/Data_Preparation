# Training and Testing Sets — Size and Composition

Project: Customer Churn Analysis (KAN-26)

## Files

| File | Description |
|---|---|
| `train_set.csv` | Training set (features + `Churn` target) |
| `test_set.csv` | Testing set (features + `Churn` target) |
| `churn_scaled.csv` | Full preprocessed dataset before splitting |

## How the split was made

- Source: the preprocessed dataset after duplicate removal (KAN-23), categorical encoding (KAN-24) and scaling (KAN-25).
- Method: `sklearn.model_selection.train_test_split`
- Ratio: 80% training / 20% testing (`test_size=0.2`)
- Stratified on `Churn` (`stratify=y`) so both sets keep the same class balance
- Reproducible: `random_state=42`

```python
X = df_scaled.drop(columns=['Churn'])
y = df_scaled['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

## Size

| Set | Rows | Share |
|---|---|---|
| Full dataset (after removing 302 duplicates from 7,043) | 6,741 | 100% |
| Training set | 5,392 | 80% |
| Testing set | 1,349 | 20% |

Each file has 11 columns: 10 features plus the `Churn` target.

## Composition

Target: `Churn` (0 = did not churn, 1 = churned).

| Set | Churn rate (share of 1s) |
|---|---|
| Full dataset | 26.6% |
| Training set | 26.6% |
| Testing set | 26.5% |

The stratified split keeps the class balance almost identical in both sets, so the test set is a fair sample of the imbalanced data (roughly 73% No / 27% Yes).

Columns in each file:

| Column | Type | Notes |
|---|---|---|
| gender | 0/1 | Female = 0, Male = 1 |
| SeniorCitizen | 0/1 | Original binary column |
| Dependents | 0/1 | No = 0, Yes = 1 |
| tenure | float | Standardised (StandardScaler) |
| PhoneService | 0/1 | No = 0, Yes = 1 |
| MultipleLines | 0/1 | No = 0, Yes = 1 |
| InternetService | 0/1 | DSL = 0, Fiber optic = 1 |
| MonthlyCharges | float | Standardised (StandardScaler) |
| Contract_One year | 0/1 | One-hot; baseline is Month-to-month |
| Contract_Two year | 0/1 | One-hot; baseline is Month-to-month |
| Churn | 0/1 | Target: No = 0, Yes = 1 |

## Notes

- Duplicates were removed before the split, so no identical row can appear in both sets.
- The scaler was fitted on the full dataset before splitting (see `Scaling_Techniques.docx`, section 6).
- Because the classes are imbalanced, use stratified cross-validation, class weighting, and metrics such as precision, recall and F1 rather than accuracy alone.
