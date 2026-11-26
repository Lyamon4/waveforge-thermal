"""Paired statistics retain missing/rejected counts and never cherry-pick tasks."""

import numpy as np
from scipy.stats import kendalltau, spearmanr


def paired_gain(rows, candidate, reference):
    lookup = {(r["task"], r["method"]): r for r in rows}
    tasks = sorted({r["task"] for r in rows})
    gains = []
    indices = []
    for task in tasks:
        a = lookup.get((task, reference))
        b = lookup.get((task, candidate))
        if a and b and a["numerical_ok"] and b["numerical_ok"] and a["peak_rise_k"] > 0:
            gains.append(100 * (a["peak_rise_k"] - b["peak_rise_k"]) / a["peak_rise_k"])
            indices.append(task)
    return dict(
        candidate=candidate,
        reference=reference,
        paired_tasks=len(gains),
        invalid_or_missing_pairs=len(tasks) - len(gains),
        task_ids=indices,
        median_gain_percent=float(np.median(gains)) if gains else None,
        wins=sum(g > 1e-9 for g in gains),
        gains_percent=gains,
    )


def rank_agreement(original, changed):
    a = {(r["task"], r["method"]): r for r in original if r["numerical_ok"]}
    b = {(r["task"], r["method"]): r for r in changed if r["numerical_ok"]}
    results = []
    for task in sorted({r["task"] for r in original}):
        methods = sorted(m for t, m in a.keys() & b.keys() if t == task)
        x = [a[task, m]["peak_rise_k"] for m in methods]
        y = [b[task, m]["peak_rise_k"] for m in methods]
        enough = len(methods) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0
        results.append(
            dict(
                task=task,
                methods=methods,
                spearman=float(spearmanr(x, y).statistic) if enough else None,
                kendall_tau_b=float(kendalltau(x, y).statistic) if enough else None,
            )
        )
    return results
