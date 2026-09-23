"""Locked Full/WISE quality replication on Recurrent-Llama-T32."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from wise.metrics import benchmark_scores
from wise.models.huginn import recurrent_token_seed
from wise.models.recurrent_llama import load_model, run_logits

METHODS = ("Full", "Static-95", "MassMatched", "SizeMatched", "WISE")


@torch.no_grad()
def generate(model, tokenizer, prompt, method, limit=32):
    active = list(tokenizer(prompt, add_special_tokens=True).input_ids)
    output = []
    stops = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    for label in ("<|end_text|>", "<|end_turn|>"):
        token = tokenizer.convert_tokens_to_ids(label)
        if isinstance(token, int) and token >= 0:
            stops.add(token)
    device = next(model.parameters()).device
    for position in range(limit):
        ids = torch.tensor([active], device=device, dtype=torch.long)
        logits = run_logits(model, ids, method, recurrent_token_seed(0, position))
        token = int(logits[0, -1].argmax())
        output.append(token); active.append(token)
        if token in stops or "\n" in tokenizer.decode(output, skip_special_tokens=True):
            break
    return tokenizer.decode(output, skip_special_tokens=True).split("\n", 1)[0].strip()


@torch.no_grad()
def gold_log_probability(model, tokenizer, prompt, gold, method):
    prefix = tokenizer(prompt, add_special_tokens=True).input_ids
    target = tokenizer(" " + gold, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix + target], dtype=torch.long, device=next(model.parameters()).device)
    logits = run_logits(model, ids, method, recurrent_token_seed(0, 10_000))[0]
    positions = torch.arange(len(prefix) - 1, len(prefix) + len(target) - 1, device=logits.device)
    tokens = torch.tensor(target, device=logits.device)
    return float(torch.log_softmax(logits[positions].float(), -1).gather(-1, tokens[:, None]).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Smoke only")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--skip-teacher", action="store_true", help="Smoke only; not full replication")
    args = parser.parse_args()
    manifest = {(r["benchmark"], r["example_id"]): r
                for r in csv.DictReader((Path(__file__).resolve().parents[1] / "manifests/recurrent_llama_primary_200.csv").open())}
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    observed = [(r["benchmark"], r["example_id"]) for r in rows]
    if len(observed) != len(set(observed)) or (args.limit is None and set(observed) != set(manifest)):
        raise ValueError("cross-backbone input IDs differ from locked N=200 cohort")
    for row in rows:
        key = row["benchmark"], row["example_id"]
        if key not in manifest or hashlib.sha256(row["prompt"].encode()).hexdigest() != manifest[key]["prompt_sha256"]:
            raise ValueError(f"locked prompt mismatch: {key}")
    if args.limit is not None:
        rows = rows[:args.limit]
    model, tokenizer = load_model(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for row in rows:
            for method in args.methods:
                answer = generate(model, tokenizer, row["prompt"], method)
                f1, em = benchmark_scores(row["benchmark"], answer, row["gold_answer"])
                result = {"benchmark": row["benchmark"], "example_id": row["example_id"],
                          "method": method, "prediction": answer, "f1": f1, "em": em}
                if not args.skip_teacher:
                    result["gold_log_probability_mean"] = gold_log_probability(
                        model, tokenizer, row["prompt"], row["gold_answer"], method)
                out.write(json.dumps(result) + "\n")
                out.flush()


if __name__ == "__main__":
    main()
