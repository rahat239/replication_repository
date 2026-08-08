import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

RANDOM_SEED = 42
df = pd.read_csv("./data/energy_regression_dataset.csv")
df["log_churn"] = np.log1p(df["net_churn"])
df["log_files"] = np.log1p(df["files_changed"])
df["log_commits_between"] = np.log1p(df["commits_between"].fillna(0))
df["add_del_ratio"] = df["lines_added"] / (df["lines_deleted"] + 1)
df["churn_per_file"] = df["net_churn"] / (df["files_changed"] + 1)
df["log_prev_energy"] = np.log1p(df["prev_energy_median"])

FEATURE_COLS = ["log_churn", "log_files", "log_commits_between", "add_del_ratio", "churn_per_file",
                 "log_prev_energy", "has_perf_keyword", "has_fix_keyword", "has_test_keyword",
                 "has_refactor_keyword", "has_dependency_keyword", "msg_total_length"]

X = df[FEATURE_COLS].fillna(0).values
y = df["is_regression"].values
groups = df["project"].values

def make_models():
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    return {
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                  random_state=RANDOM_SEED, max_depth=5, min_samples_leaf=3),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  scale_pos_weight=scale_pos_weight, random_state=RANDOM_SEED,
                                  eval_metric="logloss", use_label_encoder=False, verbosity=0),
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED,
                                                    max_iter=2000),
    }

def eval_protocol(model_name, model_fn, splitter, split_args):
    aucs, aps = [], []
    for tr, te in splitter.split(*split_args):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = model_fn()
        clf.fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], proba))
            aps.append(average_precision_score(y[te], proba))
        except ValueError:
            pass
    return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gkf = GroupKFold(n_splits=len(set(groups)))

print("=" * 90)
print(f"{'Model':<22}{'Protocol':<18}{'ROC-AUC':<20}{'PR-AUC':<20}")
print("=" * 90)

results = []
for name in ["Random Forest", "XGBoost", "Logistic Regression"]:
    for proto_name, splitter, args in [("Naive", skf, (X, y)), ("Project-grouped", gkf, (X, y, groups))]:
        model_fn = lambda n=name: make_models()[n]
        auc_m, auc_s, ap_m, ap_s = eval_protocol(name, model_fn, splitter, args)
        print(f"{name:<22}{proto_name:<18}{auc_m:.3f} +/- {auc_s:.3f}   {ap_m:.3f} +/- {ap_s:.3f}")
        results.append({"model": name, "protocol": proto_name, "roc_auc": auc_m, "roc_auc_std": auc_s,
                         "pr_auc": ap_m, "pr_auc_std": ap_s})

pd.DataFrame(results).to_csv("./data/algorithm_comparison.csv", index=False)
print("\nSaved to algorithm_comparison.csv")
