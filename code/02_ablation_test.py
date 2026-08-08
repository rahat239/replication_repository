import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
df = pd.read_csv("./data/energy_regression_dataset.csv")
df["log_churn"] = np.log1p(df["net_churn"])
df["log_files"] = np.log1p(df["files_changed"])
df["log_commits_between"] = np.log1p(df["commits_between"].fillna(0))
df["add_del_ratio"] = df["lines_added"] / (df["lines_deleted"] + 1)
df["churn_per_file"] = df["net_churn"] / (df["files_changed"] + 1)
df["log_prev_energy"] = np.log1p(df["prev_energy_median"])

FULL_FEATURES = ["log_churn", "log_files", "log_commits_between", "add_del_ratio", "churn_per_file",
                  "log_prev_energy", "has_perf_keyword", "has_fix_keyword", "has_test_keyword",
                  "has_refactor_keyword", "has_dependency_keyword", "msg_total_length"]
NO_BASELINE_FEATURES = [f for f in FULL_FEATURES if f != "log_prev_energy"]

y = df["is_regression"].values
groups = df["project"].values

def get_oof(X, splitter, split_args):
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(*split_args):
        if y[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                      random_state=RANDOM_SEED, max_depth=5, min_samples_leaf=3)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof

def bootstrap_gap(yv, p_naive, p_grouped, n_boot=2000):
    rng = np.random.RandomState(RANDOM_SEED)
    pos_idx, neg_idx = np.where(yv == 1)[0], np.where(yv == 0)[0]
    gaps = []
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        try:
            gaps.append(roc_auc_score(yv[bi], p_naive[bi]) - roc_auc_score(yv[bi], p_grouped[bi]))
        except ValueError:
            continue
    gaps = np.array(gaps)
    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    p_two_sided = 2 * min((gaps <= 0).mean(), (gaps >= 0).mean())
    return gaps.mean(), ci_lo, ci_hi, p_two_sided

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gkf = GroupKFold(n_splits=len(set(groups)))

print("=" * 70)
print("ABLATION: log_prev_energy IN vs OUT")
print("=" * 70)

for label, feat_cols in [("WITH log_prev_energy (full feature set)", FULL_FEATURES),
                          ("WITHOUT log_prev_energy (ablated)", NO_BASELINE_FEATURES)]:
    X = df[feat_cols].fillna(0).values
    oof_naive = get_oof(X, skf, (X, y))
    oof_grouped = get_oof(X, gkf, (X, y, groups))
    mask = ~np.isnan(oof_naive) & ~np.isnan(oof_grouped)
    yv, p_naive, p_grouped = y[mask], oof_naive[mask], oof_grouped[mask]
    auc_naive = roc_auc_score(yv, p_naive)
    auc_grouped = roc_auc_score(yv, p_grouped)
    gap_mean, ci_lo, ci_hi, p_val = bootstrap_gap(yv, p_naive, p_grouped)
    print(f"\n{label}")
    print(f"  Naive AUC:    {auc_naive:.3f}")
    print(f"  Grouped AUC:  {auc_grouped:.3f}")
    print(f"  Bootstrap gap: {gap_mean:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]  p={p_val:.4f}")
