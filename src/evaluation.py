"""
Statistical evaluation utilities.

Includes:
- DeLong's test for comparing ROC AUCs
- AUC difference with confidence intervals
- Youden threshold and metrics at threshold
- Decision Curve Analysis (DCA)
"""

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import roc_curve, confusion_matrix


# ─── DeLong's Test ───
# Adapted from: https://github.com/jiesihu/AUC_Delongtest__python

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return scipy.stats.norm.sf(z) * 2


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack(
        (predictions_one, predictions_two)
    )[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def delong_test(y_true, y_pred_new, y_pred_old):
    """
    DeLong test p-value (two-sided).
    - y_pred_new: improved model predictions
    - y_pred_old: baseline model predictions
    """
    y_true = np.asarray(y_true)
    y_pred_new = np.asarray(y_pred_new, dtype=float)
    y_pred_old = np.asarray(y_pred_old, dtype=float)

    if y_true.shape[0] != y_pred_new.shape[0] or y_true.shape[0] != y_pred_old.shape[0]:
        raise ValueError("y_true, y_pred_new, y_pred_old must have the same length.")
    if np.any(~np.isfinite(y_pred_new)) or np.any(~np.isfinite(y_pred_old)):
        raise ValueError("Predictions contain NaN/Inf.")
    uniq = np.unique(y_true)
    if not np.array_equal(uniq, [0, 1]) and not np.array_equal(uniq, [1, 0]):
        raise ValueError("y_true must be binary (0/1).")
    if len(uniq) < 2:
        raise ValueError("y_true must contain both positive and negative samples.")

    p = delong_roc_test(y_true, y_pred_new, y_pred_old)
    return float(np.ravel(p)[0])


def auc_diff_with_ci(y_true, y_pred_new, y_pred_old, alpha=0.05):
    """
    Compute ΔAUC = AUC(new) - AUC(old) with 95% CI using DeLong covariance.

    Returns dict with: auc_new, auc_old, delta_auc, se, ci_low, ci_high, z, p
    """
    order, label_1_count = compute_ground_truth_statistics(y_true)
    preds_sorted = np.vstack((y_pred_new, y_pred_old))[:, order]
    aucs, cov = fastDeLong(preds_sorted, label_1_count)

    auc_new, auc_old = float(aucs[0]), float(aucs[1])
    delta = auc_new - auc_old
    var = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    var = max(var, 1e-12)
    se = np.sqrt(var)

    z = delta / se
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
    z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
    ci_low = delta - z_crit * se
    ci_high = delta + z_crit * se

    return {
        "auc_new": auc_new,
        "auc_old": auc_old,
        "delta_auc": delta,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z": z,
        "p": p,
    }


# ─── Youden Threshold & Metrics ───

def youden_threshold(y_true, y_prob):
    """
    Find optimal threshold maximizing Youden's index (TPR - FPR).
    Returns: (threshold, tpr_at_threshold, fpr_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    return float(thresholds[idx]), float(tpr[idx]), float(fpr[idx])


def metrics_at_threshold(y_true, y_prob, thr):
    """
    Compute classification metrics at a given probability threshold.
    Returns dict with: threshold, sensitivity, specificity, ppv, npv, accuracy, f1, tp, tn, fp, fn
    """
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = (2 * ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
    return {
        "threshold": thr,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "accuracy": acc,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ─── Decision Curve Analysis ───

def decision_curve(y_true, y_prob, thresholds=None):
    """
    Compute net benefit across thresholds for Decision Curve Analysis.
    Returns DataFrame with columns: threshold, net_benefit
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    y_true = np.asarray(y_true).astype(int)
    N = len(y_true)
    out = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / N) - (fp / N) * (th / (1 - th))
        out.append((th, nb))
    return pd.DataFrame(out, columns=["threshold", "net_benefit"])


# ─── Plot Utilities ───

def plot_roc_curve(y_true, y_prob, title="ROC Curve", threshold_markers=None):
    """Plot ROC curve with optional threshold markers.

    Parameters
    ----------
    threshold_markers : list of dict, optional
        Each dict has keys 'thr', 'label'. The corresponding (FPR, TPR)
        point is plotted on the curve.

    Returns matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")

    if threshold_markers:
        for m in threshold_markers:
            thr = m["thr"]
            y_pred = (np.asarray(y_prob) >= thr).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            m_tpr = tp / (tp + fn + 1e-12)
            m_fpr = fp / (fp + tn + 1e-12)
            ax.scatter([m_fpr], [m_tpr], s=30, marker="o",
                       label=m.get("label", f"thr={thr:.3f}"))

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_calibration_curve(y_true, y_prob, title="Calibration Curve", n_bins=10):
    """Plot calibration curve. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve as _cal_curve
    from sklearn.metrics import brier_score_loss

    prob_true, prob_pred = _cal_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(f"{title}\nBrier={brier:.3f}")
    fig.tight_layout()
    return fig


def plot_dca(y_true, y_prob, title="Decision Curve Analysis"):
    """Plot Decision Curve Analysis. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt

    dca_df = decision_curve(y_true, y_prob)
    prev = np.asarray(y_true).mean()
    treat_all = dca_df["threshold"].apply(
        lambda th: prev - (1 - prev) * (th / (1 - th))
    )

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(dca_df["threshold"], dca_df["net_benefit"], lw=2, label="Model")
    ax.plot(dca_df["threshold"], treat_all, "--", label="Treat-all")
    ax.axhline(0, linestyle="--", color="gray", label="Treat-none")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
