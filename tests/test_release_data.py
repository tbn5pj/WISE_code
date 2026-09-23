import csv
from pathlib import Path

from wise.config import WISEConfig


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_configuration_and_locked_cohorts():
    assert WISEConfig() == WISEConfig(32, 12, 32, 0.95, (9, 10, 11, 12))
    rows = read("manifests/huginn_primary_200.csv")
    assert len(rows) == 200
    assert len({(r["benchmark"], r["example_id"]) for r in rows}) == 200
    assert sum(r["benchmark"] == "HotpotQA" for r in rows) == 100
    assert sum(r["benchmark"] == "2WikiMultihopQA" for r in rows) == 100
    assert len(read("manifests/mechanism_60.csv")) == 60
    assert len(read("manifests/context_scaling_120.csv")) == 120
    cross = read("manifests/recurrent_llama_primary_200.csv")
    assert len(cross) == 200 and len({(r["benchmark"], r["example_id"]) for r in cross}) == 200


def test_saved_attention_speedups_are_arithmetically_consistent():
    for row in read("results/canonical_scaling_summary.csv"):
        actual = float(row["Dense20_ms_median"]) / float(row["WISE20_setup_ms_median"])
        assert abs(actual - float(row["inclusive_ratio_of_medians"])) < 1e-12
    for row in read("results/figure2d_block_size.csv"):
        actual = float(row["dense_ms_median"]) / float(row["wise_ms_median"])
        assert abs(actual - float(row["speedup"])) < 1e-12
