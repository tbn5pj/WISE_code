"""Rebuild the locked HotpotQA N=30 × four-length natural context panel.

The donor ordering/packing is frozen; every resulting prompt and token count
must match its included canonical hash/length before anything is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from scripts.prepare_inputs import native_context, normalize_space, records
from wise.models.huginn import DEFAULT_MODEL, DEFAULT_REVISION


def stable_key(namespace: str, value: str) -> str:
    # Historical selection namespace is retained byte-for-byte for donor order.
    prefix = bytes.fromhex("7773612d62656e63686d61726b2d657870616e73696f6e2d7631")
    payload = prefix + b"\0" + namespace.encode() + b"\0" + value.encode()
    return hashlib.sha256(payload).hexdigest()


def gold_context(row: dict) -> str:
    required = set(row["supporting_facts"]["title"])
    context = row["context"]
    return "\n\n".join(
        f"[{title}]\n{normalize_space(' '.join(body))}"
        for title, body in zip(context["title"], context["sentences"])
        if title in required
    )


def make_prompt(tokenizer, context: str, question: str) -> str:
    user = ("Answer the question using the supplied context. Return only a concise answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def pack(tokenizer, row: dict, evidence: str, units: list[dict], limit: int) -> tuple[str, list[str], int]:
    base = make_prompt(tokenizer, evidence, row["question"])
    if len(tokenizer(base, add_special_tokens=False).input_ids) > limit:
        raise RuntimeError("required evidence exceeds target; never truncate it")
    chosen = []
    for unit in units:
        distractors = "\n\n".join(item["text"] for item in chosen + [unit])
        context = "\n\n".join(x for x in (distractors, evidence) if x)
        candidate = make_prompt(tokenizer, context, row["question"])
        count = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if count <= limit:
            chosen.append(unit)
        if count >= limit - 16:
            break
    distractors = "\n\n".join(item["text"] for item in chosen)
    context = "\n\n".join(x for x in (distractors, evidence) if x)
    result = make_prompt(tokenizer, context, row["question"])
    return result, [item["unit_id"] for item in chosen], len(tokenizer(result, add_special_tokens=False).input_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=True)
    manifest = list(csv.DictReader((root / "manifests/context_scaling_120.csv").open()))
    selected = {r["example_id"] for r in csv.DictReader((root / "manifests/huginn_primary_200.csv").open())
                if r["benchmark"] == "HotpotQA"}
    raw = records(args.hotpot_json)
    by_id = {r["id"]: r for r in raw}
    units = []
    for row in raw:
        if row["id"] in selected:
            continue
        for index, text in enumerate(native_context("HotpotQA", row).split("\n\n")):
            if text.strip():
                units.append({"unit_id": f"{row['id']}:{index}", "text": text})
    units.sort(key=lambda item: stable_key("HotpotQA:natural-donor-order", item["unit_id"]))
    verified = []
    donor_sets = {}
    for item in manifest:
        example_id = item["example_id"]
        base_id = example_id.split("__natural_", 1)[0]
        target = int(item["target_context_length"])
        limit = 4000 if target == 4096 else target
        row = by_id[base_id]
        evidence = gold_context(row)
        content, donor_ids, actual = pack(tokenizer, row, evidence, units, limit)
        if hashlib.sha256(content.encode()).hexdigest() != item["prompt_sha256"]:
            raise RuntimeError(f"locked packing hash mismatch: {example_id}")
        if actual != int(item["actual_context_prompt_length"]):
            raise RuntimeError(f"locked token length mismatch: {example_id}")
        prior = donor_sets.get(base_id, set())
        current = set(donor_ids)
        if not prior.issubset(current):
            raise RuntimeError(f"non-nested donor set: {example_id}")
        donor_sets[base_id] = current
        verified.append({"example_id": example_id, "prompt": content, "gold_answer": row["answer"]})
    if len(verified) != 120 or len(donor_sets) != 30:
        raise RuntimeError("expected exactly 30 matched examples × four contexts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for item in verified:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Verified and wrote {len(verified)} packed contexts: {args.output}")


if __name__ == "__main__":
    main()
