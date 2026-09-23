"""Render compact paper-facing tables from included finalized summary CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results"


def read(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows):
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    systems = read("canonical_scaling_summary.csv")
    block = read("figure2d_block_size.csv")
    freeze = read("freeze_a_aggregate.csv")
    truncate = read("truncate_t12_aggregate.csv")
    hotpot = read("hotpot_controls.csv")
    wiki = read("2wiki_controls.csv")
    cross = read("recurrent_llama_causal_summary.csv")
    names = {"full": "Full", "b32_static_95": "Static-95",
             "b32_static_massmatched": "MassMatched", "b32_static_sizematched": "SizeMatched",
             "b32_wise_12": "WISE"}

    def quality_rows(benchmark, controls):
        output = []
        for row in controls:
            method = names.get(row["method_id"])
            if method is not None:
                density = row.get("block_density", "") or ("1.0" if method == "Full" else "")
                future = row.get("future_full_attention_mass_retained", "") or ("1.0" if method == "Full" else "")
                output.append((benchmark, method, row.get("f1", row.get("F1")), row.get("em", row.get("EM")),
                               density, future))
        for source, method in ((freeze, "Freeze-A"), (truncate, "Truncate@12")):
            row = next(r for r in source if r["benchmark"] == benchmark and r["method"] == method)
            output.append((benchmark, method, row["F1"], row["EM"],
                           row["block_density"] or "--", row["future_mass"] or "--"))
        return output

    parts = ["# Included finalized WISE summaries", "",
             "Values below are read verbatim from the accompanying compact CSVs. They are attention-workload, not model-forward, timings.", "",
             "## Primary Huginn behavioral controls (locked N=100 per benchmark)", "",
             table(["Benchmark", "Method", "F1", "EM", "Block density", "Future mass"],
                   quality_rows("HotpotQA", hotpot) + quality_rows("2WikiMultihopQA", wiki)), "",
             "## Recurrent-Llama-T32 Full/WISE replication", "",
             table(["Benchmark", "Method", "N", "F1", "EM", "Block density", "Future mass"],
                   [(r["benchmark"], r["method"], r["N"], r["F1"], r["EM"],
                     r["block_density"], r["future_mass"]) for r in cross if r["method"] in ("Full", "WISE")]), "",
             "## Validated B32 canonical N=30 attention workload", "",
             table(["Context", "N", "Density %", "Dense 20 ms", "WISE 20 + setup ms", "Speedup"],
                   [(r["context"], r["N"], r["density_pct"], r["Dense20_ms_median"],
                     r["WISE20_setup_ms_median"], r["inclusive_ratio_of_medians"]) for r in systems]), "",
             "## Final 2K block-size control", "",
             table(["B", "Density %", "Future mass %", "Dense ms", "WISE ms", "Speedup"],
                   [(r["B"], r["block_density_percent"], r["future_mass_percent"],
                     r["dense_ms_median"], r["wise_ms_median"], r["speedup"]) for r in block]), "",
             "## Freeze-A and Truncate@12 paired differences", "",
             table(["Benchmark", "Control", "N", "F1", "EM", "F1 minus Full"],
                   [(r["benchmark"], r["method"], r["N"], r["F1"], r["EM"], r["delta_F1_vs_Full"])
                    for r in freeze + truncate if r["method"] in ("Freeze-A", "Truncate@12")]), ""]
    content = "\n".join(parts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content)


if __name__ == "__main__":
    main()
