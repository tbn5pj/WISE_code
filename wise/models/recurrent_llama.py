"""Pinned Recurrent-Llama-T32 no-cache reference adapter with native GQA/RoPE."""

from __future__ import annotations

import math
import sys

import torch

from wise.config import PAPER_CONFIG
from wise.interventions import freeze_a_weights, recurrence_one_controls
from wise.models.huginn import DiscoveryCapture, LateAttention, seed_everything
from wise.support import temporal_union


DEFAULT_MODEL = "smcleish/Recurrent-Llama-3.2-train-recurrence-32"  # Public third-party checkpoint.
DEFAULT_REVISION = "a5f6f126e9d9302445346844f1d7d29d111a06eb"


def load_model(model_id: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.test_time_noise = 0
    return model, tokenizer


def attention_components(module, x: torch.Tensor, freqs_cis):
    batch, length, width = x.shape
    q, k, v = module.Wqkv(x).split(module.chunks, dim=2)
    q = q.view(batch, length, module.n_head, module.head_dim).transpose(1, 2)
    k = k.view(batch, length, module.n_kv_heads, module.head_dim).transpose(1, 2)
    v = v.view(batch, length, module.n_kv_heads, module.head_dim).transpose(1, 2)
    if module.config.qk_bias:
        q_bias, k_bias = module.qk_bias.split(1, dim=0)
        q, k = (q + q_bias.transpose(1, 2)).to(q.dtype), (k + k_bias.transpose(1, 2)).to(k.dtype)
    remote = sys.modules[module.__class__.__module__]
    cos, sin = freqs_cis
    q, k = remote.apply_rotary_pos_emb(q, k, cos, sin)
    if module.n_head % module.n_kv_heads:
        raise ValueError("GQA head ratio is nonintegral")
    repeat = module.n_head // module.n_kv_heads
    k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(module.head_dim)
    causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
    return q, k, v, scores, causal, width


@torch.no_grad()
def run_logits(model, input_ids: torch.Tensor, method: str = "WISE", seed: int = 0) -> torch.Tensor:
    """Use the checkpoint's own prelude/core/coda; intervene only after step 12."""
    if method not in {"Full", "Static-95", "MassMatched", "SizeMatched", "WISE", "Freeze-A@12", "Truncate@12"}:
        raise ValueError(method)
    if input_ids.shape[0] != 1:
        raise ValueError("reference runner accepts one unpadded sequence")
    seed_everything(seed)
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
    embeds = model.transformer.wte(input_ids)
    if model.emb_scale != 1:
        embeds = embeds * model.emb_scale
    freqs = model.rotary_emb(embeds, positions)
    block_idx = torch.tensor(-1, device="cpu", dtype=torch.long)
    for block in model.transformer.prelude:
        block_idx += 1
        embeds = block(embeds, freqs, block_idx, None, None)
    state = model.initialize_state(embeds)
    capture = DiscoveryCapture(model, attention_components) if method not in {"Full", "Truncate@12"} else None
    stop = 12 if method == "Truncate@12" else 32
    try:
        for step in range(1, min(stop, 12) + 1):
            if capture is not None:
                capture.step = step
            state, block_idx = model.core_block_forward(state, embeds, freqs, None, None, block_idx, step - 1)
    finally:
        if capture is not None:
            capture.close()
    late = None
    if method not in {"Full", "Truncate@12"}:
        layers = len(model.transformer.core_block)
        wise = torch.stack([temporal_union({t: capture.discover[(t, layer)] for t in PAPER_CONFIG.discovery_steps})
                            for layer in range(layers)])
        if method in {"WISE", "Freeze-A@12"}:
            layout = wise
        else:
            layout = torch.stack([recurrence_one_controls(capture.step_one[layer], wise[layer])[method]
                                  for layer in range(layers)])
        frozen = torch.stack([freeze_a_weights(capture.step_twelve[layer], wise[layer])
                              for layer in range(layers)]) if method == "Freeze-A@12" else None
        late = LateAttention(model, layout, frozen, attention_components).initialize(length)
    try:
        for step in range(13, stop + 1):
            state, block_idx = model.core_block_forward(state, embeds, freqs, None, None, block_idx, step - 1)
    finally:
        if late is not None:
            late.close()
    coda_idx = torch.tensor(0, device="cpu", dtype=torch.long)
    for block in model.transformer.coda:
        coda_idx -= 1
        state = block(state, freqs, coda_idx, None, None)
    state = model.transformer.ln_f(state)
    return model.lm_head(state)
