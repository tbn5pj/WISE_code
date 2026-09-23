"""Build hash-verified HotpotQA/GSM8K diagnostic prompts from public records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

from scripts.prepare_inputs import native_context, records
from wise.models.huginn import DEFAULT_MODEL, DEFAULT_REVISION


def chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot-json", type=Path, required=True)
    parser.add_argument("--gsm8k-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=True)
    manifest = list(csv.DictReader((Path(__file__).resolve().parents[1] / "manifests/mechanism_60.csv").open()))
    hotpot = {r["id"]: r for r in records(args.hotpot_json)}
    gsm = records(args.gsm8k_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for item in manifest:
            benchmark, example_id = item["benchmark"], item["example_id"]
            if benchmark == "HotpotQA":
                row = hotpot[example_id]
                context = native_context(benchmark, row)
                content = ("Answer the question using the supplied context. Return only a concise answer.\n\n"
                           f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer:")
                answer = row["answer"]
            else:
                row = gsm[int(example_id.removeprefix("test-"))]
                content = ("Solve the problem and show your reasoning. End with `#### ` followed by the final numeric answer.\n\n"
                           f"Question: {row['question']}\nAnswer:")
                match = re.search(r"####\s*([^\n]+)\s*$", row["answer"])
                answer = match.group(1).strip() if match else ""
            prompt = chat(tokenizer, content)
            if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"]:
                raise RuntimeError(f"diagnostic prompt mismatch: {benchmark}/{example_id}")
            out.write(json.dumps({"benchmark": benchmark, "example_id": example_id,
                                  "prompt": prompt, "gold_answer": answer}) + "\n")
    print(f"Verified {len(manifest)} diagnostic prompts: {args.output}")


if __name__ == "__main__":
    main()
