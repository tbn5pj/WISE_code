"""No-cache Huginn reference runner for the seven paper interventions.

The model's native recurrent step and native post-recurrent exit are reused.
This reference path materializes probabilities for discovery and is not the
measured sparse systems implementation.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
import types

import numpy as np
import torch

from wise.config import PAPER_CONFIG
from wise.interventions import freeze_a_weights, recurrence_one_controls
from wise.support import direct_block_support, temporal_union, token_support


DEFAULT_MODEL = "tomg-group-umd/huginn-0125"  # Public third-party checkpoint.
DEFAULT_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recurrent_token_seed(base_seed: int, position: int) -> int:
    # Historical hash namespace is required for identical recurrent initialization.
    namespace = bytes.fromhex("7773612d62656e63686d61726b2d726563757272656e742d736565642d7631")
    payload = namespace + b"\0" + str(base_seed).encode() + b"\0" + str(position).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_model(model_id: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load a pinned public checkpoint; remote model code must be trusted by user."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.test_time_noise = 0
    return model, tokenizer


def attention_components(module, x: torch.Tensor, freqs_cis: torch.Tensor):
    """Reconstruct the checkpoint's native Q/K/V and FP32 scores, with RoPE."""
    batch, length, width = x.shape
    q, k, v = module.Wqkv(x).split(module.chunks, dim=2)
    q = q.view(batch, length, module.n_head, module.head_dim)
    k = k.view(batch, length, module.n_kv_heads, module.head_dim)
    v = v.view(batch, length, module.n_kv_heads, module.head_dim)
    if module.n_head != module.n_kv_heads:
        raise NotImplementedError("this Huginn checkpoint has equal Q/KV heads")
    if module.config.qk_bias:
        q_bias, k_bias = module.qk_bias.split(1, dim=0)
        q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
    remote = sys.modules[module.__class__.__module__]
    q, k = remote.apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(module.head_dim)
    causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
    return q, k, v, scores, causal, width


def projected_attention(module, probabilities: torch.Tensor, values: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    transferred = torch.matmul(probabilities.float(), values.float())
    batch, _, length, _ = transferred.shape
    return module.proj(transferred.transpose(1, 2).reshape(batch, length, width).contiguous().to(dtype))


class DiscoveryCapture:
    """Read-only capture from native Full steps 1 and 9--12, one example."""

    def __init__(self, model, components_fn=attention_components):
        self.step = 0
        self.components_fn = components_fn
        self.step_one = {}
        self.discover = {}
        self.step_twelve = {}
        self.handles = [block.attn.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                        for i, block in enumerate(model.transformer.core_block)]

    def _hook(self, layer):
        def hook(module, args, kwargs):
            if self.step not in (1, 9, 10, 11, 12):
                return
            mask = args[3] if len(args) > 3 else kwargs.get("mask")
            cache = args[4] if len(args) > 4 else kwargs.get("past_key_values")
            if mask is not None or cache is not None:
                raise NotImplementedError("release reference supports only unpadded no-cache input")
            x = args[0] if args else kwargs["x"]
            freqs = args[1] if len(args) > 1 else kwargs["freqs_cis"]
            _, _, _, scores, causal, _ = self.components_fn(module, x, freqs)
            probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), -1)[0].float()
            if self.step == 1:
                self.step_one[layer] = probabilities
            if self.step in PAPER_CONFIG.discovery_steps:
                self.discover[(self.step, layer)] = direct_block_support(probabilities)
            if self.step == 12:
                self.step_twelve[layer] = probabilities
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


class LateAttention:
    """Swap only late attention; Q/K/V and within-support weights stay fresh in WISE."""

    def __init__(self, model, layout: torch.Tensor, frozen: torch.Tensor | None = None,
                 components_fn=attention_components):
        self.model, self.frozen = model, frozen
        self.components_fn = components_fn
        self.masks = []
        self.layout = layout
        self.original = {}

    def initialize(self, sequence_length: int):
        self.masks = [token_support(item, sequence_length) for item in self.layout]
        for layer, block in enumerate(self.model.transformer.core_block):
            self.original[layer] = block.attn.forward
            block.attn.forward = types.MethodType(self._forward(layer), block.attn)
        return self

    def _forward(self, layer):
        def forward(module, x, freqs_cis, block_idx, mask=None, past_key_values=None):
            if mask is not None or past_key_values is not None:
                raise NotImplementedError("only unpadded no-cache attention is supported")
            _, _, values, scores, _, width = self.components_fn(module, x, freqs_cis)
            if self.frozen is None:
                weights = torch.softmax(scores.masked_fill(~self.masks[layer], -torch.inf), -1)
            else:
                weights = self.frozen[layer].unsqueeze(0)
            return projected_attention(module, weights, values, width, x.dtype)
        return forward

    def close(self):
        for layer, block in enumerate(self.model.transformer.core_block):
            block.attn.forward = self.original[layer]
        self.original.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@torch.no_grad()
def run_logits(model, input_ids: torch.Tensor, method: str = "WISE", seed: int = 0) -> torch.Tensor:
    """Return native-exit logits for one unpadded prompt or teacher-forced input."""
    if method not in {"Full", "Static-95", "MassMatched", "SizeMatched", "WISE", "Freeze-A@12", "Truncate@12"}:
        raise ValueError(f"unknown intervention: {method}")
    if input_ids.shape[0] != 1:
        raise ValueError("reference runner accepts batch size one")
    seed_everything(seed)
    if method == "Truncate@12":
        return model(input_ids=input_ids, num_steps=12).logits
    embeds, block_idx = model.embed_inputs(input_ids)
    state = model.initialize_state(embeds)
    capture = DiscoveryCapture(model) if method != "Full" else None
    try:
        for paper_step in range(1, 13):
            if capture is not None:
                capture.step = paper_step
            state, block_idx, _ = model.iterate_one_step(embeds, state, block_idx=block_idx, current_step=paper_step - 1)
    finally:
        if capture is not None:
            capture.close()
    if method == "Full":
        late = None
    else:
        layers = len(model.transformer.core_block)
        wise = torch.stack([temporal_union({t: capture.discover[(t, layer)] for t in PAPER_CONFIG.discovery_steps})
                            for layer in range(layers)])
        if method in {"WISE", "Freeze-A@12"}:
            layout = wise
        else:
            layout = torch.stack([
                recurrence_one_controls(capture.step_one[layer], wise[layer])[method]
                for layer in range(layers)
            ])
        frozen = torch.stack([freeze_a_weights(capture.step_twelve[layer], wise[layer])
                              for layer in range(layers)]) if method == "Freeze-A@12" else None
        late = LateAttention(model, layout, frozen).initialize(input_ids.shape[1])
    try:
        for paper_step in range(13, 33):
            state, block_idx, _ = model.iterate_one_step(embeds, state, block_idx=block_idx, current_step=paper_step - 1)
    finally:
        if late is not None:
            late.close()
    return model.predict_from_latents(state).logits


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, method: str, max_new_tokens: int = 32) -> str:
    active = list(tokenizer(prompt, add_special_tokens=False).input_ids)
    generated = []
    stops = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    for label in ("<|end_text|>", "<|end_turn|>"):
        token = tokenizer.convert_tokens_to_ids(label)
        if isinstance(token, int) and token >= 0:
            stops.add(token)
    device = next(model.parameters()).device
    for position in range(max_new_tokens):
        ids = torch.tensor([active], device=device, dtype=torch.long)
        logits = run_logits(model, ids, method, recurrent_token_seed(0, position))
        token = int(logits[0, -1].argmax())
        active.append(token)
        generated.append(token)
        if token in stops:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
