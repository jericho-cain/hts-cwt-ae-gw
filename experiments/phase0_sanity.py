#!/usr/bin/env python3
"""
Phase 0 LOSA sanity check: verify time-shift distortion is visible.

Run before integrating LOSA into training. If the overlay plot looks identical,
epsilon is too small for the chosen duration. If it looks like garbage,
interpolation may be blowing up (unlikely with linear).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import apply_losa_constant_accel, epsilon_from_accel


def main():
    T = 4096
    fs = 1024.0
    duration = T / fs

    h = generate_isolated_chirp(
        T=T, sample_rate=fs, f_start=30, f_end=250, amplitude=1.0
    )

    a = 0.5  # m/s^2 (placeholder; adjust after computing epsilon)
    eps = epsilon_from_accel(a, duration)
    print(f"duration={duration:.3f}s  a={a} m/s^2  epsilon~{eps:.3e}")

    h2 = apply_losa_constant_accel(h, sample_rate=fs, a_los=a)

    t = np.arange(T) / fs
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(t, h, label="isolated")
    axes[0].plot(t, h2, label="LOSA (const accel)", alpha=0.8)
    axes[0].legend()
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("strain (arb.)")
    axes[0].set_title("Phase 0 sanity: time shift distortion")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, h2 - h)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("residual")
    axes[1].set_title("LOSA - isolated residual")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = ROOT / "experiments" / "phase0_sanity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
