"""Plot included exact integer-step and final block-size source data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results"


def read(name):
    with (ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = read("mechanism_fraction_converged.csv")
    for benchmark in sorted({r["benchmark"] for r in curves}):
        rows = sorted((r for r in curves if r["benchmark"] == benchmark), key=lambda r: int(r["recurrence_step"]))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for signal in ("S", "A", "H", "O"):
            ax.plot([int(r["recurrence_step"]) for r in rows],
                    [float(r[f"fraction_{signal}"]) for r in rows], label=signal)
        ax.set(xlabel="Recurrent step", ylabel="Fraction converged", xlim=(1, 32), ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"mechanism_{benchmark}.png", dpi=200)
        plt.close(fig)
    block = read("figure2d_block_size.csv")
    x = [int(r["B"]) for r in block]
    fig, left = plt.subplots(figsize=(6, 3.5))
    right = left.twinx()
    left.plot(x, [float(r["block_density_percent"]) for r in block], "o-", label="Block density")
    right.plot(x, [float(r["speedup"]) for r in block], "s-", color="tab:orange", label="Dense / WISE")
    right.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    left.set(xlabel="Logical block size", ylabel="Block density (%)", xticks=x)
    right.set_ylabel("Attention-workload speedup")
    fig.tight_layout()
    fig.savefig(args.output_dir / "block_size.png", dpi=200)
    plt.close(fig)
    scaling = read("canonical_scaling_summary.csv")
    x = [int(r["context"]) for r in scaling]
    fig, left = plt.subplots(figsize=(6, 3.5))
    right = left.twinx()
    left.plot(x, [float(r["density_pct"]) for r in scaling], "o-", label="B32 density")
    right.plot(x, [float(r["inclusive_ratio_of_medians"]) for r in scaling], "s-", color="tab:orange", label="Dense / WISE")
    right.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    left.set(xlabel="Context tokens", ylabel="B32 block density (%)")
    right.set_ylabel("Attention-workload speedup")
    fig.tight_layout()
    fig.savefig(args.output_dir / "context_scaling.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
