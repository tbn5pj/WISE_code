from types import SimpleNamespace

import torch

from wise.models.huginn import attention_components, run_logits


def apply_rotary_emb_complex_like(q, k, freqs_cis=None):
    return q, k


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Wqkv = torch.nn.Linear(8, 24, bias=False)
        self.chunks = (8, 8, 8)
        self.n_head = self.n_kv_heads = 1
        self.head_dim = 8
        self.config = SimpleNamespace(qk_bias=False)
        self.proj = torch.nn.Identity()

    def forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
        _, _, v, scores, causal, width = attention_components(self, x, freqs_cis)
        weights = scores.masked_fill(~causal, -torch.inf).softmax(-1)
        return (weights @ v.float()).transpose(1, 2).reshape(x.shape[0], x.shape[1], width).to(x.dtype)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(12, 8)
        self.transformer = SimpleNamespace(core_block=[SimpleNamespace(attn=TinyAttention())])
        self.head = torch.nn.Linear(8, 12)
        self.steps = []
        self.exits = 0

    def embed_inputs(self, ids):
        return self.embed(ids), torch.tensor(0)

    def initialize_state(self, embeds):
        return torch.zeros_like(embeds)

    def iterate_one_step(self, embeds, state, block_idx, current_step):
        self.steps.append(current_step)
        x = state + embeds * 0.1
        return x + self.transformer.core_block[0].attn(x, None, block_idx), block_idx, None

    def predict_from_latents(self, state):
        self.exits += 1
        return SimpleNamespace(logits=self.head(state))

    def forward(self, input_ids, num_steps):
        embeds, idx = self.embed_inputs(input_ids)
        state = self.initialize_state(embeds)
        for step in range(num_steps):
            state, idx, _ = self.iterate_one_step(embeds, state, idx, step)
        return self.predict_from_latents(state)


def test_huginn_reference_timing_and_native_exit():
    model = TinyModel()
    ids = torch.tensor([[1, 2, 3, 4]])
    full = run_logits(model, ids, "Full")
    assert model.steps == list(range(32)) and model.exits == 1
    model.steps.clear()
    wise = run_logits(model, ids, "WISE")
    assert model.steps == list(range(32)) and model.exits == 2
    assert torch.allclose(full, wise, atol=1e-5)
    model.steps.clear()
    frozen = run_logits(model, ids, "Freeze-A@12")
    assert frozen.shape == full.shape and model.steps == list(range(32))
    model.steps.clear()
    truncated = run_logits(model, ids, "Truncate@12")
    assert truncated.shape == full.shape and model.steps == list(range(12))
    assert model.exits == 4
