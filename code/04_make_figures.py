import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import shap

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})
FIGDIR = "/home/claude/manuscript/figures"
RANDOM_SEED = 42

df = pd.read_csv("/home/claude/pipeline_test/artifacts_v2/energy_regression_dataset.csv")
thresh_df = pd.read_csv("/home/claude/pipeline_test/artifacts_v2/threshold_sensitivity.csv")
breakdown_df = pd.read_csv("/home/claude/pipeline_test/artifacts_v2/per_project_breakdown.csv")

df["log_churn"] = np.log1p(df["net_churn"])
df["log_files"] = np.log1p(df["files_changed"])
df["log_commits_between"] = np.log1p(df["commits_between"].fillna(0))
df["add_del_ratio"] = df["lines_added"] / (df["lines_deleted"] + 1)
df["churn_per_file"] = df["net_churn"] / (df["files_changed"] + 1)
df["log_prev_energy"] = np.log1p(df["prev_energy_median"])

FEATURE_COLS = ["log_churn", "log_files", "log_commits_between", "add_del_ratio", "churn_per_file",
                 "log_prev_energy", "has_perf_keyword", "has_fix_keyword", "has_test_keyword",
                 "has_refactor_keyword", "has_dependency_keyword", "msg_total_length"]
FEATURE_LABELS = {
    "log_churn": "log(net churn)", "log_files": "log(files changed)",
    "log_commits_between": "log(commits between)", "add_del_ratio": "add/delete ratio",
    "churn_per_file": "churn per file", "log_prev_energy": "log(prev. energy baseline)",
    "has_perf_keyword": "perf keyword", "has_fix_keyword": "fix keyword",
    "has_test_keyword": "test keyword", "has_refactor_keyword": "refactor keyword",
    "has_dependency_keyword": "dependency keyword", "msg_total_length": "commit msg length",
}

DUP_SIGNATURE_COLS = ["lines_added", "lines_deleted", "files_changed", "has_perf_keyword",
                       "has_fix_keyword", "has_test_keyword", "has_refactor_keyword",
                       "has_dependency_keyword"]
df["feat_signature"] = df[DUP_SIGNATURE_COLS].astype(str).agg("_".join, axis=1)
df["cluster_group"] = df["project"] + "__" + df["feat_signature"]

X = df[FEATURE_COLS].fillna(0).values
y = df["is_regression"].values
proj_groups = df["project"].values
cluster_groups = df["cluster_group"].values

# ============================================================ Fig 1: methodology pipeline diagram
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")
stages = ["EnergyTrackr\nraw measurements\n(9 projects)", "Chronological\ncommit ordering\n(git log)",
          "Statistical\nlabeling\n(Mann-Whitney U +\nCliff's delta)", "Diff-based\nfeature mining\n(git diff / log)",
          "Near-duplicate\naudit\n(signature hashing)", "3-protocol\nevaluation\n(naive / project /\ncluster-disjoint)"]
n = len(stages)
box_w, box_h, gap = 1.35, 1.15, 0.35
x0 = 0.1
for i, s in enumerate(stages):
    x = x0 + i * (box_w + gap)
    rect = plt.Rectangle((x, 0.4), box_w, box_h, facecolor="#eef3f8", edgecolor="#2c3e50", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + box_w/2, 0.4 + box_h/2, s, ha="center", va="center", fontsize=8.3, wrap=True)
    if i < n - 1:
        ax.annotate("", xy=(x + box_w + gap - 0.05, 0.4 + box_h/2), xytext=(x + box_w + 0.03, 0.4 + box_h/2),
                     arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.3))
ax.set_xlim(0, x0 + n * (box_w + gap))
ax.set_ylim(0, 2.0)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig1_pipeline.pdf", bbox_inches="tight")
plt.close()
print("Fig 1 saved.")

# ============================================================ Fig 2: per-project / per-language breakdown
fig, ax = plt.subplots(figsize=(7.5, 4))
bd = breakdown_df.sort_values("n_transitions", ascending=True)
colors = ["#4c72b0" if l == "java" else "#dd8452" for l in bd["language"]]
bars = ax.barh(bd["project"], bd["n_transitions"], color=colors, alpha=0.85, label="_nolegend_")
for i, (n_t, n_r) in enumerate(zip(bd["n_transitions"], bd["n_regressions"])):
    ax.text(n_t + 30, i, f"{n_r} reg.", va="center", fontsize=8.5, color="#333")
ax.set_xlabel("Number of commit transitions")
ax.set_title("Commit transitions and labeled energy regressions per project", fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4c72b0", label="Java"), Patch(color="#dd8452", label="Go")], loc="lower right")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig2_breakdown.pdf", bbox_inches="tight")
plt.close()
print("Fig 2 saved.")

# ============================================================ Fig 3: 3-protocol comparison (ROC-AUC, PR-AUC) + dummy
def fold_eval(splitter, split_args):
    aucs, aps = [], []
    for tr, te in splitter.split(*split_args):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                      random_state=RANDOM_SEED, max_depth=5, min_samples_leaf=3)
        clf.fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], proba)); aps.append(average_precision_score(y[te], proba))
        except ValueError:
            pass
    return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)

def dummy_eval(splitter, split_args):
    aucs = []
    for tr, te in splitter.split(*split_args):
        if y[te].sum() == 0 or y[tr].sum() == 0:
            continue
        d = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED).fit(X[tr], y[tr])
        proba = d.predict_proba(X[te])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te], proba))
        except ValueError:
            pass
    return np.mean(aucs)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gkf_proj = GroupKFold(n_splits=len(set(proj_groups)))
gkf_cluster = GroupKFold(n_splits=5)

protocols = ["Naive\n(random)", "Project-\ngrouped", "Cluster-\ndisjoint"]
auc_means, auc_stds, ap_means, ap_stds, dummy_aucs = [], [], [], [], []
for splitter, args in [(skf, (X, y)), (gkf_proj, (X, y, proj_groups)), (gkf_cluster, (X, y, cluster_groups))]:
    am, asd, apm, apsd = fold_eval(splitter, args)
    auc_means.append(am); auc_stds.append(asd); ap_means.append(apm); ap_stds.append(apsd)
    dummy_aucs.append(dummy_eval(splitter, args))

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
x = np.arange(len(protocols))
axes[0].bar(x, auc_means, yerr=auc_stds, capsize=4, color="#4c72b0", alpha=0.85, width=0.55)
axes[0].scatter(x, dummy_aucs, color="#c44e52", marker="D", s=45, zorder=5, label="Dummy baseline")
axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
axes[0].set_xticks(x); axes[0].set_xticklabels(protocols, fontsize=9.5)
axes[0].set_ylabel("ROC-AUC"); axes[0].set_ylim(0.3, 0.85)
axes[0].set_title("(a) ROC-AUC by evaluation protocol", fontsize=10.5)
axes[0].legend(fontsize=8.5, loc="upper right")

axes[1].bar(x, ap_means, yerr=ap_stds, capsize=4, color="#55a868", alpha=0.85, width=0.55)
axes[1].axhline(y.mean(), color="gray", linestyle=":", linewidth=1, label=f"Base rate ({y.mean():.3f})")
axes[1].set_xticks(x); axes[1].set_xticklabels(protocols, fontsize=9.5)
axes[1].set_ylabel("PR-AUC (Average Precision)")
axes[1].set_title("(b) PR-AUC by evaluation protocol", fontsize=10.5)
axes[1].legend(fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_protocol_comparison.pdf", bbox_inches="tight")
plt.close()
print("Fig 3 saved:", dict(zip(protocols, zip(auc_means, ap_means))))

# ============================================================ Fig 4: bootstrap gap distribution
def get_oof_predictions(splitter, split_args):
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(*split_args):
        if y[train_idx].sum() == 0:
            continue
        sc = StandardScaler().fit(X[train_idx])
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                      random_state=RANDOM_SEED, max_depth=5, min_samples_leaf=3)
        clf.fit(sc.transform(X[train_idx]), y[train_idx])
        oof[test_idx] = clf.predict_proba(sc.transform(X[test_idx]))[:, 1]
    return oof

oof_naive = get_oof_predictions(skf, (X, y))
oof_grouped = get_oof_predictions(gkf_proj, (X, y, proj_groups))
mask = ~np.isnan(oof_naive) & ~np.isnan(oof_grouped)
yv, p_naive, p_grouped = y[mask], oof_naive[mask], oof_grouped[mask]

rng = np.random.RandomState(RANDOM_SEED)
pos_idx, neg_idx = np.where(yv == 1)[0], np.where(yv == 0)[0]
gaps = []
for _ in range(2000):
    bi = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                          rng.choice(neg_idx, len(neg_idx), replace=True)])
    try:
        gaps.append(roc_auc_score(yv[bi], p_naive[bi]) - roc_auc_score(yv[bi], p_grouped[bi]))
    except ValueError:
        continue
gaps = np.array(gaps)
ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(gaps, bins=50, color="#4c72b0", alpha=0.8, edgecolor="white")
ax.axvline(0, color="gray", linestyle=":", linewidth=1.3, label="No difference")
ax.axvline(gaps.mean(), color="#c44e52", linestyle="-", linewidth=1.6, label=f"Observed gap = {gaps.mean():.3f}")
ax.axvspan(ci_lo, ci_hi, color="#c44e52", alpha=0.12, label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
ax.set_xlabel("Bootstrap ROC-AUC gap (naive − project-grouped)")
ax.set_ylabel("Bootstrap resample count")
ax.set_title("Bootstrap distribution of the naive-vs-grouped AUC gap\n(2,000 stratified resamples)", fontsize=10.5)
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig4_bootstrap_gap.pdf", bbox_inches="tight")
plt.close()
print(f"Fig 4 saved. gap mean={gaps.mean():.4f}, CI=({ci_lo:.4f},{ci_hi:.4f})")

# ============================================================ Fig 5: threshold sensitivity
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(thresh_df["threshold"]*100, thresh_df["naive_auc"], marker="o", label="Naive", color="#4c72b0", linewidth=2)
ax.plot(thresh_df["threshold"]*100, thresh_df["grouped_auc"], marker="s", label="Project-grouped", color="#c44e52", linewidth=2)
ax.fill_between(thresh_df["threshold"]*100, thresh_df["naive_auc"], thresh_df["grouped_auc"], color="gray", alpha=0.15)
ax.set_xlabel("Regression threshold (% energy increase)")
ax.set_ylabel("ROC-AUC")
ax.set_title("Naive-vs-grouped AUC gap across regression thresholds", fontsize=10.5)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_threshold_sensitivity.pdf", bbox_inches="tight")
plt.close()
print("Fig 5 saved.")

# ============================================================ Fig 6: SHAP summary
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
final_clf = RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                    random_state=RANDOM_SEED, max_depth=5, min_samples_leaf=3)
final_clf.fit(X_scaled, y)
explainer = shap.TreeExplainer(final_clf)
shap_values = explainer.shap_values(X_scaled)
# Handle both legacy (list of per-class 2D arrays) and current
# (single 3D array: samples x features x classes) SHAP output shapes.
if isinstance(shap_values, list):
    sv = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    sv = shap_values[:, :, 1]  # positive class
else:
    sv = shap_values
assert sv.shape == X_scaled.shape, f"unexpected SHAP shape {sv.shape} vs X {X_scaled.shape}"

nice_names = [FEATURE_LABELS[c] for c in FEATURE_COLS]
plt.figure(figsize=(7.5, 4.5))
shap.summary_plot(sv, X_scaled, feature_names=nice_names, show=False, plot_size=None)
plt.title("SHAP feature importance for energy-regression prediction", fontsize=10.5, pad=14)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig6_shap_summary.pdf", bbox_inches="tight")
plt.close()
print("Fig 6 saved. sv shape:", sv.shape)

# ============================================================ Fig 7: near-duplicate audit
sig_counts = df["feat_signature"].value_counts()
dup_sizes = sig_counts.values
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(dup_sizes, bins=np.arange(1, 40) - 0.5, color="#dd8452", alpha=0.85, edgecolor="white")
ax.set_xlabel("Near-duplicate signature group size")
ax.set_ylabel("Number of such groups")
ax.set_yscale("log")
pct_dup = (sig_counts[sig_counts > 1].sum() / len(df)) * 100
ax.set_title(f"Near-duplicate diff signatures across the dataset\n({pct_dup:.1f}% of all transitions belong to a group of size > 1)", fontsize=10.2)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig7_near_duplicate_audit.pdf", bbox_inches="tight")
plt.close()
print("Fig 7 saved.")

# save numeric summary for the manuscript text
summary = {
    "n_total": len(df), "n_positive": int(y.sum()), "base_rate": float(y.mean()),
    "n_projects": df["project"].nunique(), "n_java": (df["language"]=="java").sum(),
    "n_go": (df["language"]=="go").sum(),
    "naive_auc": auc_means[0], "naive_auc_std": auc_stds[0],
    "grouped_auc": auc_means[1], "grouped_auc_std": auc_stds[1],
    "cluster_auc": auc_means[2], "cluster_auc_std": auc_stds[2],
    "naive_ap": ap_means[0], "grouped_ap": ap_means[1], "cluster_ap": ap_means[2],
    "dummy_naive": dummy_aucs[0], "dummy_grouped": dummy_aucs[1], "dummy_cluster": dummy_aucs[2],
    "pooled_oof_naive": roc_auc_score(yv, p_naive), "pooled_oof_grouped": roc_auc_score(yv, p_grouped),
    "bootstrap_gap_mean": float(gaps.mean()), "bootstrap_ci_lo": float(ci_lo), "bootstrap_ci_hi": float(ci_hi),
    "near_dup_pct": float(pct_dup), "n_dup_groups": int((sig_counts>1).sum()),
    "n_cluster_groups": df["cluster_group"].nunique(),
}
import json
def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    raise TypeError
with open("/home/claude/manuscript/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=_default)
print("\nSummary:", json.dumps(summary, indent=2, default=_default))
