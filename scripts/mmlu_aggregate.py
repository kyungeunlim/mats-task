"""Aggregate MMLU per-subject accuracies from lm-eval result files.

Written by Claude Code from this prompt:

    Write scripts/mmlu_aggregate.py. It should read the three lm-eval result
    files under results/eval/{base,filtered,cb}_mmlu/ (the results*.json
    inside each), extract the per-subject acc,none and the number of samples
    per subject, and print two aggregates per model with binomial standard
    errors: (1) the full 57-subject weighted mean, which should match the
    harness's mmlu group score in the same file, and (2) the same mean
    excluding these subjects: college_biology, high_school_biology, virology,
    anatomy, clinical_knowledge, medical_genetics, college_medicine,
    professional_medicine. Output a markdown table with one row per model.

The eight excluded subjects were chosen as the biology/medical MMLU subjects
that overlap the WMDP-Bio forget domain, following point 8 of
docs/plan_review_20260826.md.

Reads the results*.json under results/eval/{base,filtered,cb}_mmlu/, computes
(1) the full 57-subject sample-weighted mean (cross-checked against the
harness's own "mmlu" group score in the same file) and (2) the same mean
excluding bio/medical-adjacent subjects. Standard errors are binomial:
sqrt(p*(1-p)/n) on the pooled correct-count.

Usage: python scripts/mmlu_aggregate.py
"""

import glob
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("base", "results/eval/base_mmlu"),
    ("filtered", "results/eval/filtered_mmlu"),
    ("cb", "results/eval/cb_mmlu"),
]

EXCLUDED_SUBJECTS = {
    "college_biology",
    "high_school_biology",
    "virology",
    "anatomy",
    "clinical_knowledge",
    "medical_genetics",
    "college_medicine",
    "professional_medicine",
}


def find_results_file(eval_dir):
    paths = glob.glob(os.path.join(REPO_ROOT, eval_dir, "*", "results_*.json"))
    if len(paths) != 1:
        sys.exit(f"Expected exactly one results_*.json under {eval_dir}, found {len(paths)}: {paths}")
    return paths[0]


def pooled_mean_stderr(pairs):
    """pairs: list of (acc, n) per subject. Returns (weighted mean, binomial SE)."""
    n_total = sum(n for _, n in pairs)
    correct = sum(acc * n for acc, n in pairs)
    p = correct / n_total
    se = math.sqrt(p * (1 - p) / n_total)
    return p, se, n_total


def aggregate(path):
    with open(path) as f:
        data = json.load(f)

    # "results" also holds the four category groups (mmlu_stem, ...); the leaf
    # subject tasks are the union of the categories' subtask lists.
    leaf_tasks = [
        task
        for category in data["group_subtasks"]["mmlu"]
        for task in data["group_subtasks"][category]
    ]

    subjects = {}  # subject name (without mmlu_ prefix) -> (acc, n)
    for task in leaf_tasks:
        res = data["results"][task]
        subject = task[len("mmlu_"):]
        acc = res["acc,none"]
        n = data["n-samples"][task]["effective"]
        assert n == res["sample_len"], f"{task}: n-samples {n} != sample_len {res['sample_len']}"
        subjects[subject] = (acc, n)

    if len(subjects) != 57:
        sys.exit(f"{path}: expected 57 subjects, found {len(subjects)}")
    missing = EXCLUDED_SUBJECTS - subjects.keys()
    if missing:
        sys.exit(f"{path}: excluded subjects not found in results: {missing}")

    full = pooled_mean_stderr(list(subjects.values()))
    kept = [v for k, v in subjects.items() if k not in EXCLUDED_SUBJECTS]
    excl = pooled_mean_stderr(kept)

    harness = data["results"]["mmlu"]["acc,none"]
    return full, excl, harness


def main():
    rows = []
    for label, eval_dir in MODELS:
        path = find_results_file(eval_dir)
        (full_p, full_se, full_n), (excl_p, excl_se, excl_n), harness = aggregate(path)

        diff = abs(full_p - harness)
        check = "match" if diff < 1e-9 else f"MISMATCH (diff={diff:.2e})"
        print(f"{label}: {os.path.relpath(path, REPO_ROOT)}")
        print(f"  full mean {full_p:.6f} vs harness mmlu {harness:.6f} -> {check}")
        print(f"  full n={full_n}, excluded-subjects n={excl_n}")

        rows.append((label, full_p, full_se, excl_p, excl_se))

    print()
    print("| Model | MMLU (57 subjects) | MMLU (excl. 8 bio/med subjects) |")
    print("|---|---|---|")
    for label, fp, fse, ep, ese in rows:
        print(f"| {label} | {fp:.4f} ± {fse:.4f} | {ep:.4f} ± {ese:.4f} |")


if __name__ == "__main__":
    main()
