import einops
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import (
  Rotary,
  EmbeddingLayer,
  LayerNorm,
  TimestepEmbedder,
  DDiTFinalLayer,
  modulate_fused,
  bias_dropout_add_scale_fused_train,
  bias_dropout_add_scale_fused_inference,
  split_and_apply_rotary_pos_emb,
  regular_attention_multi_headed,
)


class _Expert(nn.Module):
  def __init__(self, dim, mlp_ratio):
    super().__init__()
    hidden = int(mlp_ratio * dim)
    self.fc1 = nn.Linear(dim, hidden, bias=True)
    self.act = nn.GELU(approximate='tanh')
    self.fc2 = nn.Linear(hidden, dim, bias=True)

  def forward(self, x):
    return self.fc2(self.act(self.fc1(x)))


def topk_gumbel_ste_route(routing_logits, num_experts, topk, tau,
                          route_tau=1.0, deterministic=False):
  _logits = routing_logits.float()
  if route_tau == 0:
    deterministic = True
  elif route_tau != 1.0:
    _logits = _logits / route_tau
  if deterministic:
    expert_weights = F.softmax(_logits, dim=-1)
  else:
    expert_weights = F.gumbel_softmax(
        _logits, tau=tau, hard=False, dim=-1)
  sel_w, sel_e = expert_weights.topk(topk, dim=-1)
  sel_w = sel_w / (sel_w.sum(dim=-1, keepdim=True) + 1e-8)

  hard_idx = sel_w.argmax(dim=-1, keepdim=True)
  hard_w_in_topk = torch.zeros_like(sel_w).scatter_(
      -1, hard_idx, 1.0)
  hard_w = (hard_w_in_topk - sel_w).detach() + sel_w

  full_w = torch.zeros_like(routing_logits.float())
  full_w.scatter_(-1, sel_e, hard_w)
  return full_w


class MoEBlock(nn.Module):
  def __init__(self, dim, n_heads, num_experts, cond_dim,
               mlp_ratio=4, dropout=0.1, topk=2, route_tau=1.0,
               deterministic=False, sparse_infer=True):
    super().__init__()
    self.deterministic = deterministic
    self.sparse_infer = sparse_infer
    self.n_heads = n_heads
    self.num_experts = num_experts
    self.topk = topk
    self.route_tau = route_tau
    self.dim = dim

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.experts = nn.ModuleList(
      [_Expert(dim, mlp_ratio) for _ in range(num_experts)])
    self.router = nn.Linear(dim, num_experts, bias=False)
    nn.init.normal_(self.router.weight, std=0.02)

    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    return (bias_dropout_add_scale_fused_train if self.training
            else bias_dropout_add_scale_fused_inference)

  def _sparse_moe(self, h, full_w):
    B, L, D = h.shape
    hf = h.reshape(B * L, D)
    wf = full_w.reshape(B * L, -1)
    sel = wf.argmax(dim=-1)
    w_sel = wf.gather(-1, sel[:, None])
    order = torch.argsort(sel)
    counts = torch.bincount(sel, minlength=wf.shape[-1]).tolist()
    hs = hf[order]
    outs, start = [], 0
    for k, n in enumerate(counts):
      if n:
        hk = hs[start:start + n]
        if n < 64:
          hk = torch.cat([hk, hk.new_zeros(64 - n, hk.shape[1])], dim=0)
        outs.append(self.experts[k](hk)[:n])
        start += n
    ys = torch.cat(outs, dim=0)
    out = torch.empty_like(ys)
    out[order] = ys
    out = out * w_sel.to(out.dtype)
    return out.reshape(B, L, D)

  def forward(self, x, rotary_cos_sin, c, force_route_logits=None, tau=1.0):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)
    (shift_msa, scale_msa, gate_msa,
     shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(
       c)[:, None].chunk(6, dim=2)
    x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x),
      'b s (three h d) -> b s three h d',
      three=3, h=self.n_heads)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
    x = regular_attention_multi_headed(q, k, v)
    x = bias_dropout_scale_fn(
      self.attn_out(x), None, gate_msa, x_skip, self.dropout)

    h_pre_mlp = modulate_fused(self.norm2(x), shift_mlp, scale_mlp)

    self_router_logits = self.router(h_pre_mlp.float())
    routing_logits = (force_route_logits if force_route_logits is not None
                      else self_router_logits)
    full_w = topk_gumbel_ste_route(routing_logits, self.num_experts,
                                   self.topk, tau, route_tau=self.route_tau,
                                   deterministic=self.deterministic)

    if self.sparse_infer and not torch.is_grad_enabled():
      moe_out = self._sparse_moe(h_pre_mlp, full_w)
    else:
      all_outs = torch.stack(
        [e(h_pre_mlp) for e in self.experts], dim=0)
      moe_out = torch.einsum(
        "kbld,blk->bld", all_outs, full_w.to(all_outs.dtype))

    x = bias_dropout_scale_fn(moe_out, None, gate_mlp, x, self.dropout)

    return x, self_router_logits


class MoEDIT(nn.Module):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = config.algo.causal_attention
    assert not self.causal, 'MoEDIT does not support causal mode'
    self.config = config
    self.vocab_size = vocab_size

    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    num_experts = config.algo.get('num_experts', 8)
    topk = int(config.algo.get('max_expert_per_tok', 2))
    mlp_ratio = config.model.get('mlp_ratio', 4)
    self.num_experts = num_experts
    self.topk = topk

    self.route_tau = float(config.algo.get('eval_route_tau', 1.0))
    self.n_blocks = config.model.n_blocks

    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)

    self.blocks = nn.ModuleList([
      MoEBlock(
        dim=dim, n_heads=config.model.n_heads,
        num_experts=num_experts, cond_dim=cond_dim,
        mlp_ratio=mlp_ratio, dropout=config.model.dropout,
        topk=topk, route_tau=self.route_tau,
        deterministic=bool(config.algo.get('det_moe', False)),
        sparse_infer=bool(config.algo.get('sparse_moe_infer', True)))
      for _ in range(self.n_blocks)
    ])

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim, out_channels=vocab_size,
      cond_dim=cond_dim, adaLN=True)
    self.scale_by_sigma = config.model.scale_by_sigma

  def forward(self, x, sigma, class_cond=None, weights=None,
              force_route_logits=None, return_router_logits=False, tau=1.0):
    assert class_cond is None, 'Class cond not implemented for MoE-DIT'
    x = self.vocab_embed(x, weights)
    t_cond = F.silu(self.sigma_map(sigma))
    rotary_cos_sin = self.rotary_emb(x)

    all_router_logits = []
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i, block in enumerate(self.blocks):
        force_d = (force_route_logits[i]
                   if force_route_logits is not None else None)
        x, rl = block(x, rotary_cos_sin, c=t_cond,
                      force_route_logits=force_d, tau=tau)
        all_router_logits.append(rl)
      x = self.output_layer(x, c=t_cond)

    if return_router_logits:
      return x, all_router_logits
    return x
