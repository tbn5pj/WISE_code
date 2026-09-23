"""Hash-check the locked Recurrent-Llama plain-completion QA prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from scripts.prepare_inputs import native_context, records
from wise.models.recurrent_llama import DEFAULT_MODEL, DEFAULT_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot-json", type=Path, required=True)
    parser.add_argument("--2wiki-json", dest="wiki_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=True)
    manifest = list(csv.DictReader((Path(__file__).resolve().parents[1] / "manifests/recurrent_llama_primary_200.csv").open()))
    sources = {
        "HotpotQA": {row["id"]: row for row in records(args.hotpot_json)},
        "2WikiMultiHopQA": {row["_id"]: row for row in records(args.wiki_json)},
    }
    verified = []
    for item in manifest:
        benchmark, example_id = item["benchmark"], item["example_id"]
        row = sources[benchmark][example_id]
        context = native_context(benchmark, row)
        prompt = f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer:"
        length = len(tokenizer(prompt, add_special_tokens=True).input_ids)
        if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"] or length != int(item["prompt_tokens"]):
            raise RuntimeError(f"locked cross-backbone prompt mismatch: {benchmark}/{example_id}")
        verified.append({"benchmark": benchmark, "example_id": example_id,
                         "prompt": prompt, "gold_answer": row["answer"]})
    if len(verified) != 200:
        raise RuntimeError("expected N=200 locked cross-backbone prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for row in verified:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Verified {len(verified)} cross-backbone prompts: {args.output}")


if __name__ == "__main__":
    main()
