"""Rebuild and hash-check canonical Huginn prompts from separately obtained data.

Inputs are user-supplied JSON/JSONL records in the public dataset schemas.
This script never downloads or redistributes dataset content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from wise.models.huginn import DEFAULT_MODEL, DEFAULT_REVISION


def records(path: Path):
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def native_context(benchmark: str, row: dict) -> str:
    if benchmark == "HotpotQA":
        context = row["context"]
        return "\n\n".join(f"[{title}]\n{normalize_space(' '.join(body))}"
                           for title, body in zip(context["title"], context["sentences"]))
    return "\n\n".join(f"[{title}]\n{normalize_space(' '.join(body))}"
                       for title, body in row["context"])


def prompt(tokenizer, benchmark: str, row: dict) -> str:
    context = native_context(benchmark, row)
    user = ("Answer the question using the supplied context. Return only a concise answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer:")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot-json", type=Path, required=True)
    parser.add_argument("--2wiki-json", dest="wiki_json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, trust_remote_code=True)
    manifest = list(csv.DictReader((Path(__file__).resolve().parents[1] / "manifests/huginn_primary_200.csv").open()))
    sources = {
        "HotpotQA": {row["id"]: row for row in records(args.hotpot_json)},
        "2WikiMultihopQA": {row["_id"]: row for row in records(args.wiki_json)},
    }
    verified = []
    for item in manifest:
        benchmark, example_id = item["benchmark"], item["example_id"]
        row = sources[benchmark][example_id]
        content = prompt(tokenizer, benchmark, row)
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest != item["prompt_sha256"]:
            raise RuntimeError(f"canonical prompt hash mismatch: {benchmark}/{example_id}")
        verified.append({"benchmark": benchmark, "example_id": example_id,
                         "prompt": content, "gold_answer": row["answer"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in verified:
            handle.write(json.dumps(row) + "\n")
    print(f"Verified {len(manifest)} locked prompts: {args.output}")


if __name__ == "__main__":
    main()
