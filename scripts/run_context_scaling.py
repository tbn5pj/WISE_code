"""Matched Full/WISE quality evaluation on externally packed canonical contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from wise.metrics import answer_em, answer_f1
from wise.models.huginn import load_model
from scripts.run_causal_controls import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSONL with example_id,prompt,gold_answer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Smoke only")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = {r["example_id"]: r for r in csv.DictReader((root / "manifests/context_scaling_120.csv").open())}
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != len(manifest) and args.limit is None:
        raise ValueError("expected exact canonical 120 context-variant prompts")
    if len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate example ID")
    for row in rows:
        record = manifest[row["example_id"]]
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != record["prompt_sha256"]:
            raise ValueError(f"prompt mismatch: {row['example_id']}")
    if args.limit is not None:
        rows = rows[:args.limit]
    model, tokenizer = load_model(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for row in rows:
            for method in ("Full", "WISE"):
                answer = generate(model, tokenizer, row["prompt"], method, 32)
                out.write(json.dumps({"example_id": row["example_id"], "context": manifest[row["example_id"]]["target_context_length"],
                                      "method": method, "answer": answer,
                                      "f1": answer_f1(answer, row["gold_answer"]),
                                      "em": answer_em(answer, row["gold_answer"])}) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
