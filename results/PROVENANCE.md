# Included result provenance

This repository contains compact, finalized paper summaries and locked cohort manifests.
The table below identifies each **included release file** by its scientific role and SHA-256 digest of the exact bytes in this archive.
It intentionally omits original research-directory paths, machine-specific run names, and internal artifact locations; none are needed to use or verify the released files.
These digests verify the integrity of the released files, not the identity or location of upstream private artifacts.

CSV numeric and other field values were copied without rounding or recomputation.
Text line endings were normalized to LF where necessary.
`paper_numbers.csv` omits GPU UUID/index columns; its retained field values are unchanged.
No raw model outputs, model checkpoints, or new timing measurements are included.

## Released numerical summaries

| Release file (under `results/`) | Measurement / role | Released-file SHA-256 |
|---|---|---|
| `paper_numbers.csv` | Finalized paper attention-timing figures; GPU-identifying columns excluded. | `25f52603ed886835335742406f080c9ae9bdedc00840ce932eafd975e9790a67` |
| `canonical_scaling_summary.csv` | Attention-kernel timing and support density across context lengths. | `c84f963a81038f1daeaf8ac7652b9916ebcdf3e8c246a4df3aa05f312b10923a` |
| `real80_summary.csv` | Captured 80-call attention-workload timing summary. | `7db2170e28e6fe4e752d965e2b3bc555f97e46beb40e69f78e46c2d5626cc0ce` |
| `matched_backend_summary.csv` | Full-support versus WISE-support timing under a matched B32 backend. | `a7e3aa1c2137063e013ed2ce175a7a7e6bd482a18fc7fb7b92f2f0991f937cab` |
| `t32_summary.csv` | Complete 32-step attention-trajectory timing summary. | `888696d6f1824fef040288896c2f772c1d0cec8fa0edbca98957b120f4f241c7` |
| `figure2d_block_size.csv` | Block-size sweep data used for the systems visualization. | `a5c260d79b388fb6d6c06a8ecce25ec372b5a4eb6db13b715220f39990db78d2` |
| `appendix_c6.csv` | Block-size comparison across downstream quality, support structure, and systems measurements. | `43106d7ce1624eb240b1c0f2dee124a958eb8b18c3c17fca215888e960d90367` |
| `hotpot_controls.csv` | Primary-backbone HotpotQA matched behavioral controls. | `e5d025a6636d7fa59037043f767b392cfbfca01f39e4837c732221dc3381f331` |
| `2wiki_controls.csv` | Primary-backbone 2WikiMultihopQA matched behavioral controls. | `a98b6b62621250c4b5d0303f2ae6a346b76cc5efbcede3c9ab8ba3e61f2374a3` |
| `freeze_a_aggregate.csv` | Matched-support fixed-attention intervention summary. | `28d9a9473c83217d833a5be819afccf05466266ebeaf5fc6e6a212a16cbfa653` |
| `truncate_t12_aggregate.csv` | Recurrent truncation at discovery depth: matched intervention summary. | `0308a2a1c2ce82df69a22973c74f41732d29c63762f1029055da7326ee323631` |
| `context_scaling.csv` | Matched HotpotQA quality and support measurements across context lengths. | `e288e30a54379cd16bdedb9bfc15e0fe7952258202c9fda4b4de6ce36753a94c` |
| `recurrent_llama_causal_summary.csv` | Second-backbone matched behavioral replication summary. | `47c3a7bb97f65a6827cdfb9d262c637fbfa58e7704d3627b96763676495f2857` |
| `recurrent_llama_mechanism_summary.csv` | Second-backbone routing and representation diagnostic summary. | `3ec35669e626dcf19b8f697f3a026236f5a144439f25a4346d613d1cb0f36733` |
| `mechanism_fraction_converged.csv` | Per-step mechanism convergence fractions. | `a1083ffb7f83dbafda2e3ca901f36cb37a6e96156618cd721b3b6069fb41c17d` |
| `mechanism_default_convergence.csv` | Per-example mechanism convergence diagnostics under the default criteria. | `2ed4da17ed782771db100a990d0362ecadf965506a61b7ba402b125c441a7166` |

## Locked cohort manifests

These manifests contain benchmark names, example identifiers, canonical order, prompt SHA-256 hashes, and/or prompt lengths as applicable; they do not redistribute source passages.

| Release file (under `manifests/`) | Cohort / role | Released-file SHA-256 |
|---|---|---|
| `huginn_primary_200.csv` | Matched primary-backbone behavioral cohort (100 HotpotQA and 100 2Wiki). | `dde10ef7df16f2afd7d7e28f24f79f56890469c7cc922409eced1eb44d017267` |
| `mechanism_60.csv` | Matched primary-backbone mechanism cohort (30 HotpotQA and 30 GSM8K). | `b8b01814c24a5500325285d76c97e9919b271753723efc4ce1cf1decb7eca6db` |
| `context_scaling_120.csv` | Context-scaling cohort: 30 fixed examples at each of four context lengths. | `d8416cabcfee2a66d992a6c9888f74f70720376fc130f6f7d27ce20184fc7956` |
| `recurrent_llama_primary_200.csv` | Matched second-backbone behavioral cohort (100 HotpotQA and 100 2Wiki). | `c8ffedbd1e3436f9953f39026310fc57c7ec4a1392ce4dffab2b656546104f8a` |

During release preparation, public-dataset and tokenizer reconstruction verified all 200 primary-backbone behavioral prompt hashes, all 60 mechanism prompt hashes, all 200 second-backbone behavioral prompt hashes and token lengths, and all 120 context-scaling prompt hashes and token lengths.
This was offline prompt preparation, not a fresh model-inference run.
The full original source passages are intentionally not bundled.

## Frozen sparse-attention implementation

The included Ampere arithmetic source `wise/kernels/_ampere_winner.py` is byte-identical to its final validated implementation (SHA-256 `d066af2c2d28a34290cd2179c354df8144e578b1099fd521dfeb0872acfa1b46`).
The included schedule builder `wise/kernels/gpu_schedule.py` is byte-identical to its validated implementation (SHA-256 `fac6fee6a7ab61fdc6bb8273a2ee0d79fc1505e8c6b67594f183f9a7f24c280c`).
The release wrapper is tested separately and does not modify either frozen source file.
