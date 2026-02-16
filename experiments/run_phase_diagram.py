#!/usr/bin/env python3
"""
Variability–Detectability Phase Diagram

Runs experiments at multiple intrinsic-spread levels and plots AUROC vs spread.
Sweeps: very_tight (jitter 0) → tight → medium → broad.
Produces the key figure: AUROC(Δφ=1), AUROC(Δφ=3) vs intrinsic spread.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Spread levels: (label, chirp_jitter_pct, tight_chirp)
# jitter_pct applies ±X% to center of each param when tight_chirp
# "broad" = full original ranges
SPREAD_LEVELS = [
    ("very_tight", 0.0, True),
    ("tight", 0.01, True),
    ("medium", 0.05, True),
    ("wide", 0.10, True),
    ("very_wide", 0.15, True),
    ("broad", None, False),
]

BASE_CONFIG = ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml"
OUTPUT_DIR = ROOT / "experiments/outputs/phase_diagram"


def run_one(spread_label: str, jitter_pct: float | None, tight_chirp: bool) -> dict | None:
    """Run experiment and return phase0_summary.json contents."""
    cfg = load_yaml(BASE_CONFIG)
    save_dir = f"experiments/outputs/phase_diagram/{spread_label}"
    cfg["experiment"]["save_dir"] = save_dir
    cfg["synthetic"]["tight_chirp"] = tight_chirp
    if tight_chirp and jitter_pct is not None:
        cfg["synthetic"]["chirp_jitter_pct"] = jitter_pct
    if not tight_chirp:
        cfg.setdefault("phase0_losa", {})["chirp_f_start"] = [10.0, 15.0]
        cfg.setdefault("phase0_losa", {})["chirp_f_end"] = [55.0, 70.0]

    # Write temp config
    tmp_cfg = ROOT / f"experiments/configs/_phase_diagram_{spread_label}.yaml"
    import yaml
    with open(tmp_cfg, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [sys.executable, "-m", "experiments.run_experiment", "--config", str(tmp_cfg)]
    logger.info(f"Running {spread_label}: jitter={jitter_pct}, tight={tight_chirp}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        logger.error(f"Failed {spread_label}: {result.stderr[:500]}")
        return None

    summary_path = ROOT / save_dir / "phase0_summary.json"
    if not summary_path.exists():
        logger.error(f"No phase0_summary.json for {spread_label}")
        return None
    with open(summary_path) as f:
        return json.load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Variability-Detectability phase diagram")
    parser.add_argument("--quick", action="store_true", help="Use quick config (2 epochs, 200 train)")
    parser.add_argument("--skip-run", action="store_true", help="Skip runs; plot from existing outputs")
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated subset, e.g. very_tight,broad")
    args = parser.parse_args()

    levels = SPREAD_LEVELS
    if args.levels:
        want = set(s.strip() for s in args.levels.split(","))
        levels = [x for x in SPREAD_LEVELS if x[0] in want]
        if not levels:
            logger.error(f"No matching levels for {args.levels}")
            return 1

    if args.quick:
        # Override for quick test: smaller n, fewer epochs
        global BASE_CONFIG
        BASE_CONFIG = ROOT / "experiments/configs/ground_phase0_tight_chirp_quick.yaml"
        logger.warning("Using quick config - results not publication-quality")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for label, jitter, tight in levels:
        save_dir = OUTPUT_DIR / label
        summary_path = save_dir / "phase0_summary.json"
        if args.skip_run and summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            results.append((label, jitter, tight, data))
            continue
        if not args.skip_run:
            data = run_one(label, jitter, tight)
            if data:
                results.append((label, jitter, tight, data))

    if not results:
        logger.error("No results to plot")
        return 1

    # Extract AUROC @ Δφ=1, Δφ=3 for each spread

    labels = [r[0] for r in results]
    x_pos = np.arange(len(labels))
    auroc_1 = []
    auroc_3 = []
    for _, _, _, d in results:
        auc = d.get("auroc_vs_bin", {})
        auroc_1.append(auc.get(1.0, auc.get("1.0", auc.get(1, np.nan))))
        auroc_3.append(auc.get(3.0, auc.get("3.0", auc.get(3, np.nan))))

    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos - width/2, auroc_1, width, label="AUROC @ Δφ=1")
    ax.bar(x_pos + width/2, auroc_3, width, label="AUROC @ Δφ=3")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Intrinsic spread")
    ax.set_title("LOSA Detectability vs Intrinsic Chirp Variability")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_ylim(0.35, 1.0)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "phase_diagram_auroc_vs_spread.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out_path}")

    # Save numeric results
    table = {
        "spread": labels,
        "jitter_pct": [r[1] for r in results],
        "tight_chirp": [r[2] for r in results],
        "auroc_dphi_1": auroc_1,
        "auroc_dphi_3": auroc_3,
    }
    with open(OUTPUT_DIR / "phase_diagram_results.json", "w") as f:
        json.dump(table, f, indent=2)
    logger.info(f"Saved {OUTPUT_DIR / 'phase_diagram_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
