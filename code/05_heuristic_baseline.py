"""
Training-free heuristic baseline: rank transitions by raw net code churn
(classic code-churn-based defect-proneness heuristic, Nagappan & Ball, ICSE 2005).
Requires no fitting, so it cannot leak by construction -- evaluated identically
under both protocols using the same fold structure as the fitted ML models.
"""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_SEED = 42
df = pd.read_csv("./data/energy_regression_dataset.csv")
y = df["is_regression"].values
groups = df["project"].values
churn_score = df["net_churn"].values

def eval_fixed_score(score, splitter, split_args):
    aucs, aps = [], []
    for tr, te in splitter.split(*split_args):
        if y[te].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y[te], score[te]))
            aps.append(average_precision_score(y[te], score[te]))
        except ValueError:
            pass
    return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gkf = GroupKFold(n_splits=len(set(groups)))

for name, splitter, args in [("Naive", skf, (churn_score, y)),
                               ("Project-grouped", gkf, (churn_score, y, groups))]:
    auc_m, auc_s, ap_m, ap_s = eval_fixed_score(churn_score, splitter, args)
    print(f"Churn-magnitude heuristic -- {name}: "
          f"ROC-AUC={auc_m:.3f}+/-{auc_s:.3f}  PR-AUC={ap_m:.3f}+/-{ap_s:.3f}")

print(f"\nWhole-dataset: ROC-AUC={roc_auc_score(y, churn_score):.3f}  "
      f"PR-AUC={average_precision_score(y, churn_score):.3f}")
