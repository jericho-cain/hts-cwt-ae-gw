"""
Phase 0 LOSA evaluation reports: histograms, AUROC, Cohen's d, detection at 1% FPR.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two samples."""
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0
    return float((m2 - m1) / pooled)


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """TPR at the threshold where FPR equals target_fpr."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right")
    if idx >= len(tpr):
        return float(tpr[-1]) if len(tpr) else 0.0
    return float(tpr[idx - 1])


def save_phase0_reports(cfg: dict, recon_stats: dict) -> None:
    """Save phase0_hist.png, phase0_auroc.png, phase0_summary.json."""
    save_dir = cfg["experiment"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    iso_errs = recon_stats["isolated"]["errs"]
    use_delta_phi = any(
        recon_stats[k].get("delta_phi", 0) > 0 for k in recon_stats if k != "isolated"
    )

    x_vals = []
    auc_vals = []
    x_label = "delta_phi (rad)" if use_delta_phi else "epsilon"

    plt.figure(figsize=(8, 5))
    plt.hist(iso_errs, bins=50, alpha=0.5, label="isolated", density=True)

    for key, val in recon_stats.items():
        if key == "isolated":
            continue
        x_val = val.get("delta_phi", val.get("eps", 0))
        errs = val["errs"]

        label = f"{x_val:.2g}" if x_val >= 0.01 else f"{x_val:.1e}"
        plt.hist(errs, bins=50, alpha=0.5, label=label, density=True)

        y = np.concatenate([np.zeros_like(iso_errs), np.ones_like(errs)])
        s = np.concatenate([iso_errs, errs])
        auc = roc_auc_score(y, s)

        x_vals.append(x_val)
        auc_vals.append(auc)

    plt.legend()
    plt.xlabel("Normalized reconstruction error")
    plt.ylabel("Density")
    plt.title("Phase 0 LOSA Reconstruction Error")
    plt.savefig(os.path.join(save_dir, "phase0_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    if x_vals:
        order = np.argsort(x_vals)
        x_sorted = np.array(x_vals)[order]
        auc_sorted = np.array(auc_vals)[order]

        plt.figure(figsize=(6, 4))
        plt.plot(x_sorted, auc_sorted, marker="o")
        if not use_delta_phi and min(x_vals) > 0:
            plt.xscale("log")
        plt.xlabel(x_label)
        plt.ylabel("AUROC")
        plt.title("Phase 0 LOSA AUROC vs " + x_label)
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(save_dir, "phase0_auroc.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    auc_map = {}
    cohens_d_map = {}
    tpr_at_1pct_fpr_map = {}

    for key, val in recon_stats.items():
        if key == "isolated":
            continue
        errs = val["errs"]
        x_val = val.get("delta_phi", val.get("eps", 0))

        y = np.concatenate([np.zeros_like(iso_errs), np.ones_like(errs)])
        s = np.concatenate([iso_errs, errs])
        auc_map[float(x_val)] = float(roc_auc_score(y, s))
        cohens_d_map[float(x_val)] = _cohens_d(iso_errs, errs)
        tpr_at_1pct_fpr_map[float(x_val)] = _tpr_at_fpr(y, s, target_fpr=0.01)

    summary = {
        key: {
            "eps": float(val.get("eps", 0)),
            "delta_phi": float(val.get("delta_phi", 0)),
            "mean_err": float(np.mean(val["errs"])),
            "std_err": float(np.std(val["errs"])),
        }
        for key, val in recon_stats.items()
    }
    summary["auroc_vs_bin"] = auc_map
    summary["cohens_d_vs_bin"] = cohens_d_map
    summary["tpr_at_1pct_fpr"] = tpr_at_1pct_fpr_map

    # Latent Mahalanobis metrics (when available)
    has_latent = all(recon_stats[k].get("latent_scores") is not None for k in recon_stats)
    if has_latent:
        iso_latent = recon_stats["isolated"]["latent_scores"]
        auc_map_latent = {}
        cohens_d_map_latent = {}
        tpr_map_latent = {}
        for key, val in recon_stats.items():
            if key == "isolated":
                continue
            scores = val["latent_scores"]
            x_val = val.get("delta_phi", val.get("eps", 0))
            y = np.concatenate([np.zeros_like(iso_latent), np.ones_like(scores)])
            s = np.concatenate([iso_latent, scores])
            auc_map_latent[float(x_val)] = float(roc_auc_score(y, s))
            cohens_d_map_latent[float(x_val)] = _cohens_d(iso_latent, scores)
            tpr_map_latent[float(x_val)] = _tpr_at_fpr(y, s, target_fpr=0.01)
        summary["auroc_latent_vs_bin"] = auc_map_latent
        summary["cohens_d_latent_vs_bin"] = cohens_d_map_latent
        summary["tpr_at_1pct_fpr_latent"] = tpr_map_latent
        dphi_l = [float(v.get("delta_phi", v.get("eps", 0))) for k, v in recon_stats.items() if k != "isolated"]
        mean_l = [float(np.mean(recon_stats[k]["latent_scores"])) for k in recon_stats if k != "isolated"]
        if len(dphi_l) >= 2:
            rho_l, p_l = spearmanr(dphi_l, mean_l)
            summary["spearman_rho_latent"] = float(rho_l)
            summary["spearman_pval_latent"] = float(p_l)
        else:
            summary["spearman_rho_latent"] = None
            summary["spearman_pval_latent"] = None

    # Spearman rho: delta_phi (or eps) vs mean recon error across LOSA bins
    dphi_vals = []
    mean_err_vals = []
    for key, val in recon_stats.items():
        if key == "isolated":
            continue
        dphi_vals.append(float(val.get("delta_phi", val.get("eps", 0))))
        mean_err_vals.append(float(np.mean(val["errs"])))
    if len(dphi_vals) >= 2:
        rho, pval = spearmanr(dphi_vals, mean_err_vals)
        summary["spearman_rho_delta_phi_vs_mean_err"] = float(rho)
        summary["spearman_pval"] = float(pval)
    else:
        summary["spearman_rho_delta_phi_vs_mean_err"] = None
        summary["spearman_pval"] = None

    with open(os.path.join(save_dir, "phase0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
