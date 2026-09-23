"""Recompute the diagnostic Top-16 S/A/H/O trajectory on locked prompts.

This diagnostic is NOT the deployed direct-B32 WISE support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wise.diagnostics import causal_top16, first_three_transition_run, jaccard, representation_threshold
from wise.models.huginn import attention_components, load_model, seed_everything


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    mid = 0.5 * (p.float() + q.float())
    return 0.5 * ((p * (p.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum(-1)
                  + (q * (q.clamp_min(1e-8).log() - mid.clamp_min(1e-8).log())).sum(-1))


def relative_l2(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    return (current.float() - previous.float()).norm(dim=-1) / previous.float().norm(dim=-1).clamp_min(1e-8)


class Observer:
    def __init__(self, model):
        self.probabilities = {}
        self.outputs = {}
        self.handles = [block.attn.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                        for i, block in enumerate(model.transformer.core_block)]

    def _hook(self, layer):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["x"]
            freqs = args[1] if len(args) > 1 else kwargs["freqs_cis"]
            _, _, v, scores, causal, _ = attention_components(module, x, freqs)
            probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), -1)[0]
            self.probabilities[layer] = probabilities.detach()
            self.outputs[layer] = (probabilities.unsqueeze(0) @ v.float())[0].detach()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def trajectory(model, ids: torch.Tensor):
    seed_everything(0)
    embeds, idx = model.embed_inputs(ids)
    state = model.initialize_state(embeds)
    previous_state, previous_a, previous_o = state, {}, {}
    records = []
    observer = Observer(model)
    try:
        for step in range(1, 33):
            observer.probabilities, observer.outputs = {}, {}
            state, idx, _ = model.iterate_one_step(embeds, state, block_idx=idx, current_step=step - 1)
            if len(observer.probabilities) != len(model.transformer.core_block):
                raise RuntimeError("incomplete attention capture")
            h = float(relative_l2(previous_state, state).mean())
            values = {"S": [], "A": [], "O": []}
            for layer, current in observer.probabilities.items():
                if layer in previous_a:
                    values["S"].append(float(jaccard(causal_top16(previous_a[layer]), causal_top16(current)).mean()))
                    values["A"].append(float(js_divergence(previous_a[layer], current).mean()))
                    values["O"].append(float(relative_l2(previous_o[layer], observer.outputs[layer]).mean()))
            records.append({"paper_step": step, "S": float(np.mean(values["S"])) if values["S"] else float("nan"),
                            "A": float(np.mean(values["A"])) if values["A"] else float("nan"),
                            "H": h, "O": float(np.mean(values["O"])) if values["O"] else float("nan")})
            previous_state = state
            previous_a = observer.probabilities
            previous_o = observer.outputs
    finally:
        observer.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSONL with benchmark, example_id, prompt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only")
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / "manifests/mechanism_60.csv"
    expected = {(r["benchmark"], r["example_id"]): r["prompt_sha256"] for r in csv.DictReader(manifest.open())}
    inputs = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    for row in inputs:
        key = row["benchmark"], row["example_id"]
        if key not in expected or hashlib.sha256(row["prompt"].encode()).hexdigest() != expected[key]:
            raise ValueError(f"noncanonical diagnostic prompt: {key}")
    observed = [(r["benchmark"], r["example_id"]) for r in inputs]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate diagnostic example ID")
    if args.limit is None and set(observed) != set(expected):
        raise ValueError("expected exact locked N=60 diagnostic cohort")
    if args.limit is not None:
        inputs = inputs[:args.limit]
    model, tokenizer = load_model(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        for row in inputs:
            ids = torch.tensor([tokenizer(row["prompt"], add_special_tokens=False).input_ids], device=args.device)
            curve = trajectory(model, ids)
            values = {key: np.asarray([r[key] for r in curve]) for key in ("S", "A", "H", "O")}
            tau = {"S": first_three_transition_run(values["S"], threshold=0.90, higher=True, first_step=1)}
            tau.update({key: first_three_transition_run(values[key], threshold=representation_threshold(values[key]),
                                                         higher=False, first_step=2) for key in ("A", "H", "O")})
            output.write(json.dumps({"benchmark": row["benchmark"], "example_id": row["example_id"],
                                     "tau": tau, "curve": curve}) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
