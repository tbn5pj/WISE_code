"""Run the seven matched Huginn controls on externally supplied locked prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from wise.metrics import benchmark_scores
from wise.models.huginn import load_model, recurrent_token_seed, run_logits


METHODS = ("Full", "Static-95", "MassMatched", "SizeMatched", "Freeze-A@12", "Truncate@12", "WISE")


def canonical_rows(inputs: Path, manifest: Path):
    expected = {(row["benchmark"], row["example_id"]): row for row in csv.DictReader(manifest.open())}
    rows = [json.loads(line) for line in inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len({(r["benchmark"], r["example_id"]) for r in rows}):
        raise ValueError("duplicate input IDs")
    for row in rows:
        key = row["benchmark"], row["example_id"]
        if key not in expected:
            raise ValueError(f"ID not in locked manifest: {key}")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != expected[key]["prompt_sha256"]:
            raise ValueError(f"locked prompt hash mismatch: {key}")
    if set((r["benchmark"], r["example_id"]) for r in rows) != set(expected):
        raise ValueError("input IDs do not equal locked canonical cohort")
    return rows


@torch.no_grad()
def generate(model, tokenizer, prompt: str, method: str, limit: int) -> str:
    active = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    output = []
    stops = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    for label in ("<|end_text|>", "<|end_turn|>"):
        token = tokenizer.convert_tokens_to_ids(label)
        if isinstance(token, int) and token >= 0:
            stops.add(token)
    device = next(model.parameters()).device
    for position in range(limit):
        ids = torch.tensor([active], dtype=torch.long, device=device)
        scores = run_logits(model, ids, method, recurrent_token_seed(0, position))
        token = int(scores[0, -1].argmax())
        active.append(token)
        output.append(token)
        if token in stops:
            break
    return tokenizer.decode(output, skip_special_tokens=True).strip()


@torch.no_grad()
def gold_log_probability(model, tokenizer, prompt: str, gold: str, method: str) -> float:
    prefix = tokenizer(prompt, add_special_tokens=False).input_ids
    target = tokenizer(" " + gold, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + target], device=next(model.parameters()).device)
    logits = run_logits(model, ids, method, recurrent_token_seed(0, 10_000))[0]
    positions = torch.arange(len(prefix) - 1, len(prefix) + len(target) - 1, device=logits.device)
    targets = torch.tensor(target, device=logits.device)
    return float(torch.log_softmax(logits[positions].float(), -1).gather(-1, targets[:, None]).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSONL from prepare_inputs.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-teacher", action="store_true", help="Smoke test only; not full paper reproduction")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "manifests/huginn_primary_200.csv"
    rows = canonical_rows(args.inputs, manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    model, tokenizer = load_model(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            for method in args.methods:
                prediction = generate(model, tokenizer, row["prompt"], method, args.max_new_tokens)
                f1, em = benchmark_scores(row["benchmark"], prediction, row["gold_answer"])
                result = {"benchmark": row["benchmark"], "example_id": row["example_id"],
                          "method": method, "prediction": prediction,
                          "f1": f1, "em": em}
                if not args.skip_teacher:
                    result["gold_log_probability_mean"] = gold_log_probability(
                        model, tokenizer, row["prompt"], row["gold_answer"], method)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
    print(f"Wrote {len(rows)} examples × {len(args.methods)} conditions to {args.output}")


if __name__ == "__main__":
    main()
