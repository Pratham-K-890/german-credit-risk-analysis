# German Credit Risk Analysis

Credit risk modelling on the UCI Statlog German Credit dataset (1,000 applicants, 20 features, ~30% default rate). Builds and compares a pruned CART decision tree against logistic regression, then audits the tree for reason-code stability, demographic fairness (disparate impact), and profit-optimal decision thresholds.

## Contents

- `notebook/german_credit_risk_analysis.ipynb` — end-to-end analysis: EDA, feature engineering, cost-complexity pruning, model comparison, fairness audit, bootstrap stability check, and profit-threshold optimization
- `dataset/` — raw and decoded UCI German Credit data
- `charts/` — exported figures from the notebook

## Key techniques

- CART decision tree with cost-complexity pruning (CCP), tuned via cross-validated ROC-AUC
- Decision tree vs. logistic regression comparison (ROC-AUC, KS statistic, F1, accuracy)
- Bootstrap stability testing of tree splits
- Disparate impact / four-fifths rule fairness audit across gender and age
- Profit-optimal classification threshold based on approval/default economics
