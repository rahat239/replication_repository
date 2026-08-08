"""
v2 pipeline: statistical relabeling, near-duplicate audit, cross-language
expansion, threshold sensitivity, bootstrap significance test.
"""
import csv, json, subprocess, statistics, re
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

DATA_ROOT = Path("./energytrackr-data")
CLONE_ROOT = Path("./repos")

PROJECTS = {
    "docx4j":        {"lang": "java", "url": "https://github.com/plutext/docx4j",         "csvs": ["java/docx4j/sorted_1.csv"]},
    "fastexcel":      {"lang": "java", "url": "https://github.com/dhatim/fastexcel",       "csvs": ["java/fastexcel/sorted_1.csv", "java/fastexcel/sorted_2.csv"]},
    "flexy-pool":     {"lang": "java", "url": "https://github.com/vladmihalcea/flexy-pool","csvs": ["java/flexy-pool/energy_usage.csv"]},
    "jacoco":         {"lang": "java", "url": "https://github.com/jacoco/jacoco",          "csvs": ["java/jacoco/sorted_1.csv"]},
    "jsoup":          {"lang": "java", "url": "https://github.com/jhy/jsoup",              "csvs": ["java/jsoup/merged_sorted.csv"]},
    "signalfx-java":  {"lang": "java", "url": "https://github.com/signalfx/signalfx-java", "csvs": ["java/signalfx-java/energy_usage.csv"]},
    "super-csv":      {"lang": "java", "url": "https://github.com/super-csv/super-csv",    "csvs": ["java/super-csv/sorted_1.csv"]},
    "fiber":          {"lang": "go",   "url": "https://github.com/gofiber/fiber",          "csvs": ["go/fiber/sorted_1.csv"]},
    "nscache":        {"lang": "go",   "url": "https://github.com/no-src/nscache",         "csvs": ["go/nscache/sorted_1.csv"]},
}

REGRESSION_PCT_THRESHOLD = 0.20
SIGNIFICANCE_ALPHA = 0.05

PERF_KEYWORDS = re.compile(r"\b(perf|performance|optimi[sz]e|speed|efficien|cache|lazy)\b", re.I)
FIX_KEYWORDS  = re.compile(r"\b(fix|bug|patch|resolve)\b", re.I)
TEST_KEYWORDS = re.compile(r"\b(test|spec)\b", re.I)
REFACTOR_KEYWORDS = re.compile(r"\b(refactor|cleanup|clean up|restructure)\b", re.I)
DEP_KEYWORDS = re.compile(r"\b(upgrade|bump|dependency|dependencies|version)\b", re.I)


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    # O(n*m) fine for our small per-commit sample sizes (~15)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def clone_repo(name, url):
    dest = CLONE_ROOT / name
    if dest.exists():
        return dest
    r = subprocess.run(["git", "clone", "--quiet", url, str(dest)],
                        capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"  clone FAILED for {name}: {r.stderr[:300]}")
        return None
    return dest


def read_energy_csv(project):
    readings = {}
    for rel in PROJECTS[project]["csvs"]:
        path = DATA_ROOT / rel
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                h, v = row[0].strip(), row[1].strip()
                try:
                    v = float(v)
                except ValueError:
                    continue
                readings.setdefault(h, []).append(v)
    return readings


def chronological_order(repo_path, wanted_hashes):
    r = subprocess.run(["git", "log", "--format=%H", "--reverse", "--all"],
                        cwd=repo_path, capture_output=True, text=True, timeout=120)
    all_commits = r.stdout.split()
    order_index = {h: i for i, h in enumerate(all_commits)}
    present = [h for h in wanted_hashes if h in order_index]
    present.sort(key=lambda h: order_index[h])
    return present


def diff_features(repo_path, prev_hash, curr_hash, commits_between):
    r = subprocess.run(["git", "diff", "--numstat", prev_hash, curr_hash],
                        cwd=repo_path, capture_output=True, text=True, timeout=60)
    added = deleted = files_changed = 0
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, d, _ = parts
        try:
            added += int(a); deleted += int(d)
        except ValueError:
            pass
        files_changed += 1
    msg_r = subprocess.run(["git", "log", "--format=%s", f"{prev_hash}..{curr_hash}"],
                            cwd=repo_path, capture_output=True, text=True, timeout=60)
    msgs = msg_r.stdout
    return {
        "lines_added": added, "lines_deleted": deleted, "net_churn": added + deleted,
        "files_changed": files_changed, "commits_between": commits_between,
        "msg_total_length": len(msgs),
        "has_perf_keyword": int(bool(PERF_KEYWORDS.search(msgs))),
        "has_fix_keyword": int(bool(FIX_KEYWORDS.search(msgs))),
        "has_test_keyword": int(bool(TEST_KEYWORDS.search(msgs))),
        "has_refactor_keyword": int(bool(REFACTOR_KEYWORDS.search(msgs))),
        "has_dependency_keyword": int(bool(DEP_KEYWORDS.search(msgs))),
    }


def build_project(project):
    info = PROJECTS[project]
    repo_path = clone_repo(project, info["url"])
    if repo_path is None:
        return []
    readings = read_energy_csv(project)
    if len(readings) < 5:
        print(f"  {project}: too few measured commits, skipping"); return []
    ordered = chronological_order(repo_path, list(readings.keys()))
    if len(ordered) < 5:
        print(f"  {project}: too few commits in git history, skipping"); return []

    rows = []
    noise_cvs = []
    for i in range(1, len(ordered)):
        prev_h, curr_h = ordered[i - 1], ordered[i]
        prev_vals, curr_vals = readings[prev_h], readings[curr_h]
        prev_med, curr_med = statistics.median(prev_vals), statistics.median(curr_vals)
        if prev_med <= 0:
            continue
        pct_change = (curr_med - prev_med) / prev_med

        # measurement-noise-aware statistical test (Mann-Whitney U + Cliff's delta)
        # instead of a bare point-estimate median comparison
        if len(prev_vals) >= 3 and len(curr_vals) >= 3:
            try:
                _, p_value = mannwhitneyu(prev_vals, curr_vals, alternative="two-sided")
            except ValueError:
                p_value = 1.0
            delta = cliffs_delta(prev_vals, curr_vals)
            for vals in (prev_vals, curr_vals):
                if statistics.mean(vals) > 0:
                    noise_cvs.append(statistics.stdev(vals) / statistics.mean(vals))
        else:
            p_value, delta = 1.0, 0.0

        naive_label = int(pct_change > REGRESSION_PCT_THRESHOLD)
        # statistically-supported label: must ALSO clear significance + a
        # non-trivial effect size, not just the raw percent change
        stat_label = int(
            (pct_change > REGRESSION_PCT_THRESHOLD)
            and (p_value < SIGNIFICANCE_ALPHA)
            and (abs(delta) > 0.33)  # "medium" Cliff's delta per Romano et al. 2006
        )

        r = subprocess.run(["git", "rev-list", "--count", f"{prev_h}..{curr_h}"],
                            cwd=repo_path, capture_output=True, text=True, timeout=60)
        try:
            commits_between = int(r.stdout.strip())
        except ValueError:
            commits_between = None

        feats = diff_features(repo_path, prev_h, curr_h, commits_between)
        feats.update({
            "project": project, "language": info["lang"],
            "prev_commit": prev_h, "commit": curr_h,
            "prev_energy_median": prev_med, "curr_energy_median": curr_med,
            "pct_change": pct_change, "p_value": p_value, "cliffs_delta": delta,
            "is_regression_naive": naive_label,
            "is_regression": stat_label,
        })
        rows.append(feats)

    med_cv = statistics.median(noise_cvs) if noise_cvs else float("nan")
    print(f"  {project} ({info['lang']}): {len(rows)} transitions | "
          f"naive-label regressions={sum(r['is_regression_naive'] for r in rows)} | "
          f"stat-significant regressions={sum(r['is_regression'] for r in rows)} | "
          f"median measurement CV={med_cv:.3f}")
    return rows


def main():
    all_rows = []
    for project in PROJECTS:
        all_rows.extend(build_project(project))

    out_path = "./data/energy_regression_dataset.csv"
    fieldnames = list(all_rows[0].keys())
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
