# WISE: Working-Set Inference for Recurrent Language Models

Reference code for *Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models*. It contains reference intervention code, the frozen exact-B32 Ampere attention kernel, locked cohort identifiers and prompt hashes, tests, and compact finalized paper summaries. It does **not** contain model weights, licensed dataset contents, raw attention tensors, or a model-level speedup claim.

WISE runs 32 recurrent steps. Steps 1–12 use unrestricted attention. At steps 9–12, it directly ranks **B32 block masses** per recurrent layer, head, and query block and selects the smallest stable cumulative-mass prefix reaching 95%. The union of those four block supports is fixed from step 13 onward. Hidden states, Q/K/V, attention weights within the selected support, and all recurrent refinement continue to update. This is neither token-support rounding nor recurrence truncation.

## Contents

- `wise/support.py`, `wise/routing.py`: exact direct block-space discovery and support freeze.
- `wise/interventions.py`: Static-95, MassMatched, SizeMatched, Freeze-A@12 and Truncate@12 reference rules.
- `wise/models/`: pinned Huginn and Recurrent-Llama-T32 no-cache reference adapters.
- `wise/kernels/`: byte-identical validated Ampere B32 arithmetic kernel, GPU CSR builder, and a minimal public wrapper.
- `scripts/`: prompt preparation, full behavioral/diagnostic runners, attention-only benchmark, and table/figure reconstruction.
- `manifests/`: IDs, prompt hashes and context lengths, not dataset text.
- `results/`: compact finalized CSVs copied without numeric changes; `results/PROVENANCE.md` identifies each source.
- `tests/` and `tools/`: lightweight correctness checks.

## Install and quick checks

Use Python 3.10+ and an environment with a CUDA-capable PyTorch build if running models or the sparse kernel. CPU-only support tests do not need a GPU.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test,plots]'
python -m pytest -q tests
python examples/minimal_wise_example.py
python -m scripts.reproduce_tables
python -m scripts.reproduce_figures --output-dir outputs/figures
```

The optional `systems` extra installs Triton. The *validated paper timing stack* was Python 3.10.20, PyTorch 2.14.0+cu126, Triton 3.8.0, NVIDIA driver 550.54.14 and RTX A6000 (SM86). A generic install may not reproduce those kernel latencies; record both dense and sparse measurements in the **same** stack and on an uncontended GPU. The included summary CSVs report the validated measurements; new timing runs may vary with hardware and software. The optional GPU unit test skips if CUDA or validated Triton is absent.

## Models and data

The two public third-party checkpoint IDs and immutable revisions are in `configs/`. The primary backbone is Huginn at T=32; the cross-backbone replication uses Recurrent-Llama-T32 without retuning WISE. Loading uses `trust_remote_code=True`; review the checkpoint's remote code and license before execution. Checkpoints and tokenizer files are downloaded by the Hugging Face libraries into *your* configured cache, never from a personal path in this repository.

Acquire HotpotQA **distractor/validation**, the official April 2021 2WikiMultihopQA **dev** set, and GSM8K **main/test** from their public distributors under their own terms. The original HotpotQA validation Arrow snapshot had SHA-256 `3ae5e9add57ca3d6d03db385981ae15d8b4f9c854372080156a7a6c6ca9b8934`; the official 2Wiki dev JSON had `79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5`; and the GSM8K test Arrow snapshot had `45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95`. An equivalent JSON export will have a different *file* hash but must preserve records/order. The preparation commands accept a Hotpot JSON/JSONL export with the public `id`, `question`, `answer`, `context.title`, `context.sentences` fields; the official 2Wiki `dev.json` list with `_id`, `question`, `answer`, `context`; and GSM8K test records in original order. The scripts reconstruct the native prompt template and **require exact SHA-256 equality** with the locked manifest. A mismatch stops the run instead of silently changing the cohort. We do not redistribute source passages. `scripts/prepare_inputs.py` verifies 100 HotpotQA + 100 2Wiki prompts; `scripts/prepare_mechanism_inputs.py` verifies 30 HotpotQA + 30 GSM8K diagnostic prompts.

```bash
python -m scripts.prepare_inputs --hotpot-json /path/to/hotpot_validation.jsonl --2wiki-json /path/to/2wiki_dev.json --output outputs/primary_200.jsonl
python -m scripts.prepare_mechanism_inputs --hotpot-json /path/to/hotpot_validation.jsonl --gsm8k-json /path/to/gsm8k_test.jsonl --output outputs/mechanism_60.jsonl
```

The paths above are illustrative user-supplied data locations, not machine-specific defaults. Exact source snapshots should be recorded by the reproducer. If a newer public dataset/tokenizer revision changes a prompt, the hash guard will flag it.

## Behavioral and mechanism reproduction

The following are **full model runs**, not smoke tests. Each generated token performs the configured recurrent computation; these jobs are expensive. Both controls and Full use the same prompt, seed namespace, greedy decoding and 32-token limit. The `--limit 1 --skip-teacher` combination is a smoke test only and is not a paper result.

```bash
python -m scripts.run_causal_controls --inputs outputs/primary_200.jsonl --output outputs/causal_controls.jsonl --methods Full Static-95 MassMatched SizeMatched Freeze-A@12 Truncate@12 WISE
python -m scripts.run_quality --inputs outputs/primary_200.jsonl --output outputs/quality.jsonl --methods Full WISE
python -m scripts.run_mechanism --inputs outputs/mechanism_60.jsonl --output outputs/mechanism_trajectories.jsonl
```

Mechanism support is **causal token Top-min(16, i+1)**, not WISE's deployed B32 support. Its convergence criterion is consecutive Jaccard ≥0.90 over three transitions. A/H/O use their own 5%-of-early-scale threshold for three transitions. These measurements diagnose the mechanism; they do not choose `t_d=12`.

The controls are intentionally distinct. Static-95 selects a step-1 direct block support but does not intervene until step 13. MassMatched and SizeMatched use the step-1 ranking with the final WISE working-set mass or local cardinality, respectively. Freeze-A@12 restricts and renormalizes the **post-softmax step-12** distribution to the exact WISE support, then uses it with fresh V(t) during steps 13–32. Truncate@12 performs all of step 12, skips step 13 onward, and keeps the checkpoint's native exit path; support density and future mass are not applicable.

The Recurrent-Llama adapter in `wise/models/recurrent_llama.py` exposes the same seven `run_logits` conditions with its checkpoint's RoPE/GQA layout. Its historical replication used a separate benchmark-native **plain-completion** prompt format, not the Huginn chat prompts. The included cross-backbone manifest locks the same 100+100 IDs but different prompt hashes; the preparer verifies all 200 hashes and token lengths before the Full/WISE replication:

```bash
python -m scripts.prepare_recurrent_llama_inputs --hotpot-json /path/to/hotpot_validation.jsonl --2wiki-json /path/to/2wiki_dev.json --output outputs/recurrent_llama_200.jsonl
python -m scripts.run_recurrent_llama_quality --inputs outputs/recurrent_llama_200.jsonl --output outputs/recurrent_llama_quality.jsonl
```

By default this reproduces Full, Static-95, MassMatched, SizeMatched and WISE with the same frozen WISE parameters; `--methods Full WISE` limits the run to the primary replication contrast.

## Context scaling

`manifests/context_scaling_120.csv` records 30 locked examples at each of 512/1K/2K/4K, including each packed prompt's SHA-256. The full packed passages are not bundled. The preparation script rebuilds the historical globally ordered natural-donor packing from the public HotpotQA validation records, preserves gold evidence, uses a 4000-token cap for the 4K point, and verifies **all 120** prompt hashes and lengths. Then run:

```bash
python -m scripts.prepare_context_scaling_inputs --hotpot-json /path/to/hotpot_validation.jsonl --output outputs/locked_contexts.jsonl
python -m scripts.run_context_scaling --inputs outputs/locked_contexts.jsonl --output outputs/context_scaling.jsonl
```

The evaluator checks all 120 prompt hashes again before model evaluation. If a different public dataset/tokenizer export prevents an exact hash match, stop and obtain the pinned source; do not treat different packing as the paper cohort. The included exact summary and plots can be reproduced without rerunning models.

## Ampere sparse-attention benchmark

The optional systems fixture is a user-supplied saved `torch` dictionary with BF16 `q`, `k`, `v` tensors `[N,55,96]` and an exact causal boolean B32 layout `[4,55,ceil(N/32),ceil(N/32)]`. The fixture is not bundled because canonical Q/K/V tensors are large. Use only a verified saved fixture for a paper comparison. Do not substitute random Q/K/V and label it canonical. Run the numerical GPU test first and verify GPU idleness independently.

```bash
python -m pip install -e '.[systems,test]'
python -m pytest -q tests/test_sparse_attention.py
python -m scripts.benchmark_attention --fixture /path/to/verified_fixture.pt --output outputs/attention_passes.csv --gpu 0 --passes 10
```

The B32 path uses GPU-resident CSR plus within-head row ordering, skips inactive key blocks, and fuses QK, causal masking, online softmax and PV. The validated arithmetic source hash is checked by a test. The benchmark covers **attention calls only**; one-time schedule construction is included only in its `setup_inclusive` condition. The included finalized CSVs also distinguish prepared, setup-inclusive, real captured 80-call, matched-backend, and full T=32 **attention trajectory** measurements. Algorithmic work reduction, kernel latency, and whole-model latency are different quantities. No end-to-end Huginn speedup is asserted here.

## Tables, plots and result provenance

`python -m scripts.reproduce_tables` prints a compact selection of included exact values. `python -m scripts.reproduce_figures --output-dir outputs/figures` plots the saved integer-step convergence curves and final 2K block-size panel; neither script alters the manuscript. For the full numeric source, use the CSVs in `results/` directly.

