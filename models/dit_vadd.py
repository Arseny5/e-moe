import math
import typing
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
      self.cos_cached[:, :, 2, :, :].fill_(1.)
      self.sin_cached[:, :, 2, :, :].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
  sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim

  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, rotary_cos_sin, c, seqlens=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    if c.ndim == 2:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c.ndim == 3:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=2)

    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    qkv = rearrange(qkv, 'b s ... -> (b s) ...')
    if seqlens is None:
      cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, step=seq_len,
        dtype=torch.int32, device=qkv.device)
    else:
      cu_seqlens = seqlens.cumsum(-1)
    x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      qkv, cu_seqlens, seq_len, 0., causal=False)

    x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x


class MaskAdaLN(nn.Module):
  def __init__(self, dim, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 3 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, x, mask_vector):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    if mask_vector.ndim == 2:
      (shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(
        mask_vector)[:, None].chunk(3, dim=2)
    elif mask_vector.ndim == 3:
      (shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(
        mask_vector).chunk(3, dim=2)

    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()

  def forward(self, x, c):
    if c.ndim == 2:
      shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    elif c.ndim == 3:
      shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)

    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class GenDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):

  def __init__(self, config, vocab_size: int, mask_index: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    if config.algo.latentMLP:
      self.latent_emb = nn.Sequential(
        nn.Linear(config.algo.latent_dim, config.algo.latent_dim),
        nn.SiLU(),
        nn.Linear(config.algo.latent_dim, config.algo.latent_dim),
        nn.SiLU(),
        nn.Linear(config.algo.latent_dim, config.model.cond_dim)
      )
    else:
      self.latent_emb = nn.Identity()

    assert config.algo.latent_struc == 'adaln'
    self.full_hidden_size = config.model.hidden_size

    self.config = config
    self.vocab_size = vocab_size
    self.mask_index = mask_index

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, indices, z, sigma):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))
    z = self.latent_emb(z)

    if z.ndim == 3:
      c = c.unsqueeze(1) + z
    else:
      c = c + z

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
      x = self.output_layer(x, c)

    return x


class InferDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size
    self.latent_dim = config.algo.latent_dim

    self.vocab_embed = EmbeddingLayer(config.decoder.hidden_size,
                                      vocab_size)
    self.sigma_map = TimestepEmbedder(config.decoder.cond_dim)
    self.rotary_emb = Rotary(
      config.decoder.hidden_size // config.decoder.n_heads)

    blocks = []
    for _ in range(config.decoder.n_blocks):
      blocks.append(DDiTBlock(config.decoder.hidden_size,
                              config.decoder.n_heads,
                              config.decoder.cond_dim,
                              dropout=config.decoder.dropout))
    self.blocks = nn.ModuleList(blocks)

    if config.decoder.mask_struc == 'indepadaln':
      indepadalns = []
      for _ in range(config.decoder.n_blocks):
        indepadalns.append(MaskAdaLN(
          config.decoder.hidden_size,
          config.decoder.cond_dim,
          dropout=config.decoder.dropout))
      self.indepadalns = nn.ModuleList(indepadalns)

    self.output_layer = DDitFinalLayer(
      config.decoder.hidden_size,
      self.latent_dim * 2,
      config.decoder.cond_dim)
    self.scale_by_sigma = config.decoder.scale_by_sigma

    if self.config.decoder.mask_struc in ['adaln', 'indepadaln']:
      self.mask_vector = nn.Parameter(
        torch.randn(config.decoder.cond_dim)
        / math.sqrt(config.decoder.cond_dim))
    elif self.config.decoder.mask_struc == 'none':
      pass
    else:
      raise NotImplementedError(
        f"The decoder mask structure '{config.decoder.mask_struc}' "
        "is not implemented.")

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def forward(self, indices, sigma, mask):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))

    if self.config.decoder.mask_struc == 'adaln':
      mask_vector = mask.to(c.dtype).unsqueeze(-1) * self.mask_vector
      c = c.unsqueeze(1) + mask_vector
    elif self.config.decoder.mask_struc == 'indepadaln':
      mask_vector = mask.to(c.dtype).unsqueeze(-1) * self.mask_vector

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)
        if self.config.decoder.mask_struc == 'indepadaln':
          x = self.indepadalns[i](x, mask_vector)
      x = self.output_layer(x, c)

    return x
