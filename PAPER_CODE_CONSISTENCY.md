# Paper/code consistency audit

| Check | Status | Release evidence |
|---|---|---|
| T=32, discovery depth 12, B=32, eta=.95, steps 9–12 | PASS (static) | `wise/config.py`, both YAML configs |
| Direct block-space sorted cumulative-mass support | PASS (unit) | `wise/support.py`, `tests/test_support.py`; no token-support-then-rounding |
| Union of steps 9–12; freeze support only after step 12 | PASS (unit) | `wise/routing.py`, `wise/models/huginn.py`; paper step 12 is `current_step=11` |
| Dynamic H/Q/K/V and within-support weights in WISE | PASS (reference-code/unit) | `LateAttention` reconstructs current Q/K/V/scores each late call; `tests/test_interventions.py` |
| Static-95: step-1 support, intervention after step 12 | PASS (reference-code) | `recurrence_one_controls`, Huginn `run_logits` |
| MassMatched: step-1 ranking, matched step-1 WISE mass | PASS (reference-code/unit) | `recurrence_one_controls` |
| SizeMatched: step-1 ranking, exact local cardinality | PASS (unit) | `recurrence_one_controls`; cardinality assertion |
| Freeze-A@12: same union support, restricted/renormalized A(12), current V | PASS (unit) | `freeze_a_weights`, `LateAttention`; row-sum test |
| Truncate@12: step 12 executes, step 13 does not, native exit | PASS (reference-code/unit) | Huginn `model(..., num_steps=12)`; `truncate_recurrence` test |
| Diagnostic token Top-16 distinct from deployed B32 | PASS (unit/static) | `wise/diagnostics.py`, `scripts/run_mechanism.py` |
| Cross-backbone same WISE parameters, native RoPE/GQA | PASS (static/input-hash audit) | `wise/models/recurrent_llama.py`, cross-backbone config; 200 locked prompt hashes verified |
| Context-scaling cohort/packing | PASS (input-hash audit) | `scripts/prepare_context_scaling_inputs.py`; 120 saved prompt hashes and token lengths reproduced |
| Exact fixed-mask sparse arithmetic; no dense fallback | PASS (source hash; GPU test conditional) | frozen `_ampere_winner.py`; `test_sparse_attention.py` |
| No unsupported model-level speedup claim | PASS (documentation scan) | README scopes systems numbers to attention workloads |
| No legacy research terminology in public text | PASS (text scan) | release public names use WISE and paper control names |

Static/unit checks do not replace a fresh full-checkpoint parity run. Large canonical model/dataset evaluations and GPU timing were **not rerun** during packaging; their saved verified result tables are copied unchanged. The exact validated kernel source and schedule builder are byte-identical to their frozen snapshots.
