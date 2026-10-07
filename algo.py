import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

import models.dit_moe
import models.dit_vadd
import models.ema
import trainer_base
import utils


class AR(trainer_base.TrainerBase):
  def __init__(self, config, tokenizer):
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert not self.config.algo.time_conditioning

  def _process_model_input(self, x0, valid_tokens):
    input_tokens = x0[:, :-1]
    output_tokens = x0[:, 1:]
    valid_tokens = valid_tokens[:, 1:]
    return input_tokens, output_tokens, valid_tokens

  def nll(self, input_tokens, labels, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del labels, current_accumulation_step, train_mode
    output = self.backbone(input_tokens, None)
    output[:, :, self.mask_index] = self.neg_infinity
    output = output.log_softmax(-1)
    return - output.gather(
      -1, output_tokens[:, :, None])[:, :, 0]

  def generate_samples(self, num_samples, **kwargs):
    # precompute token buffer
    num_pred_tokens = self.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, self.vocab_size))
             .to(self.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    for i in range(num_pred_tokens):
      output = self.backbone(x[:, :i + 1], None)
      output[:, :, self.mask_index] = self.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    return x

  def _process_sigma(self, sigma):
    del sigma
    return None


class MDLM(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    assert self.sampler != 'ancestral', \
      'sampling.predictor=ancestral is not desirable because ' \
      'it is slow. Please set sampling.predictor=ancestral_cache'

  def _process_model_output(self, model_output, xt, sigma):
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the model_output such that x.exp() is
    # a probability distribution over vocab_size.
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    del xt
    log_p_theta = torch.gather(
      input=log_x_theta,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    return log_p_theta * dalpha_t / (1 - alpha_t)

  def _get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    # score(x, t) = p_t(y) / p_t(x)
    # => log score(x, t) = log p_t(y) - log p_t(x)
    
    # case 1: x = masked
    #   (i) y = unmasked
    #     log score(x, t) = log p_\theta(x)|_y + log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    #   (ii) y = masked
    #     log score(x, t) = 0

    # case 2: x = unmasked
    #   (i) y != masked, y != x
    #     log score(x_i, t) = - inf
    #   (ii) y = x 
    #     log score(x_i, t) = 0
    #   (iii) y = masked token
    #     log score(x_i, t) = - log k
    #     where k = exp(- sigma) / (1 - exp(- sigma))
    
    log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
    assert log_k.ndim == 1
    
    masked_score = model_output + log_k[:, None, None]
    masked_score[:, :, self.mask_index] = 0

    unmasked_score = self.neg_infinity * torch.ones_like(
      model_output)
    unmasked_score = torch.scatter(
      unmasked_score,
      -1,
      x[..., None],
      torch.zeros_like(unmasked_score[..., :1]))
    unmasked_score[:, :, self.mask_index] = - (
      log_k[:, None] * torch.ones_like(x))
    
    masked_indices = (x == self.mask_index).to(
      model_output.dtype)[:, :, None]
    model_output = (
      masked_score * masked_indices
      + unmasked_score * (1 - masked_indices))
    return model_output.exp()


class SEDDAbsorb(trainer_base.AbsorbingState):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self._validate_configuration()

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampling.predictor == 'analytic'

  def _get_score(self, x, sigma):
    return self.forward(x, sigma).exp()

  def _process_model_output(self, model_output, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(model_output.dtype)
    # logits shape
    # (batch_size, context_length, vocab_size)
    model_output = (model_output
                    - esigm1_log[:, None, None]
                    - np.log(model_output.shape[-1] - 1))
    # The below scatter operation sets the log score
    # for the input word to 0.
    model_output = torch.scatter(
      model_output, -1, xt[..., None],
      torch.zeros_like(model_output[..., :1]))
    return model_output

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                    dalpha_t, low_var=False):
    """Computes the SEDD loss for the Absorbing State Diffusion.

    Args:
      log_x_theta: float torch.Tensor with shape (batch_size,
          context_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          context_length), input.
      x0: int torch.Tensor with shape (batch_size,
          context_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      alpha_t: float torch.Tensor with shape (batch_size, 1),
          signal level.
      dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
          time derivative of signal level.
      low_var: bool, low variance loss during training.
    
    Returns:
      loss with shape (batch_size, context_length).
    """
    assert not low_var
    masked_indices = xt == self.mask_index
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_x_theta[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_x_theta[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return dsigma * entropy


class EMoE(trainer_base.AbsorbingState):
  """Enhanced Mixture-of-Experts (E-MoE)."""

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    del self.backbone
    self.backbone = models.dit_moe.MoEDIT(
      self.config, vocab_size=self.vocab_size)
    self.det_moe = bool(config.algo.get('det_moe', False))
    self.kl_weight = float(config.algo.kl_weight)
    self.lb_weight = float(config.algo.lb_weight)
    self.num_experts = int(config.algo.num_experts)
    self.gumbel_tau_init = float(config.algo.gumbel_tau_init)
    self.gumbel_tau_min = float(config.algo.gumbel_tau_min)
    self.gumbel_tau_anneal_rate = float(config.algo.gumbel_tau_anneal_rate)
    self._reset_latest()
    # Re-build the EMA over the MoE backbone.
    if self.ema is not None:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    self._validate_configuration()

  def _reset_latest(self):
    self._latest_kl = None
    self._latest_lb = None
    self._latest_hprior = None
    self._latest_live = None

  def _validate_configuration(self):
    assert self.sampler != 'ancestral', \
      'sampling.predictor=ancestral is slow; use ancestral_cache'
    super()._validate_configuration()

  # MDLM parameterization of the denoiser.
  def _process_model_output(self, model_output, xt, sigma):
    return MDLM._process_model_output(self, model_output, xt, sigma)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t,
                    low_var=False):
    return MDLM.nll_per_token(
      self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=low_var)

  def _get_score(self, x, sigma):
    return MDLM._get_score(self, x, sigma)

  def _current_tau(self):
    tau = self.gumbel_tau_init * math.exp(
      -self.gumbel_tau_anneal_rate * int(self.global_step))
    return max(self.gumbel_tau_min, tau)

  def _forward_with_routes(self, x_idx, sigma, force_routes=None):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      return self.backbone(
        x=x_idx, sigma=sigma, force_route_logits=force_routes,
        return_router_logits=True, tau=self._current_tau())

  def forward(self, xt, sigma, labels=None, weights=None, nn_input_idxs=None):
    """Sampling: routes from the prior p(z | x_t)."""
    raw, _ = self._forward_with_routes(xt, sigma)
    return self._process_model_output(raw, xt, sigma)

  def _compute_kl(self, q_logits_list, p_logits_list):
    """KL(q || p) between the router distributions, summed over blocks."""
    kl_total = 0
    for ql, pl in zip(q_logits_list, p_logits_list):
      log_q = F.log_softmax(ql, dim=-1)
      log_p = F.log_softmax(pl, dim=-1)
      kl_total = kl_total + (log_q.exp() * (log_q - log_p)).sum(dim=-1)
    return kl_total                                    # (B, L)

  def _compute_lb_aux(self, router_logits_list):
    """Switch-style load-balancing loss, averaged over blocks."""
    K = self.num_experts
    lb_sum = 0.0
    for rl in router_logits_list:
      probs = F.softmax(rl, dim=-1)
      P_e = probs.mean(dim=(0, 1))
      f_e = F.one_hot(probs.argmax(-1), K).to(probs.dtype).mean(dim=(0, 1))
      lb_sum = lb_sum + K * (P_e * f_e).sum()
    return lb_sum / max(len(router_logits_list), 1)

  def _router_stats(self, router_logits_list):
    h, live = 0, 0.0
    for rl in router_logits_list:
      lp = F.log_softmax(rl.float(), dim=-1)
      p = lp.exp()
      h = h + (-(p * lp).sum(dim=-1))
      u = p.mean(dim=(0, 1))
      live += float(torch.exp(-(u * torch.log(u + 1e-12)).sum()).detach())
    return h, live / max(len(router_logits_list), 1)

  def nll(self, x0, labels, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    if self.T > 0:
      t = (t * self.T).to(torch.int) / self.T + (1 / self.T)
    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    sigma_t = self._sigma_from_alphat(alpha_t)
    xt = self.q_xt(x0, alpha_t)
    self._latest_t = t.detach()
    low_var = train_mode and self.loss_type == 'low_var'

    if (not train_mode) and bool(
        self.config.algo.get('eval_prior_routes', False)):
      raw, _ = self._forward_with_routes(xt, sigma_t)
      log_x_theta = self._process_model_output(raw, xt, sigma_t)
      self._reset_latest()
      return self.nll_per_token(
        log_x_theta=log_x_theta, xt=xt, x0=x0,
        alpha_t=alpha_t, dalpha_t=dalpha_t, low_var=False)

    if self.det_moe:
      # MDLM-MoE: deterministic routes from x_t, no posterior, no KL.
      raw, router_logits = self._forward_with_routes(xt, sigma_t)
      log_x_theta = self._process_model_output(raw, xt, sigma_t)
      recon_per_tok = self.nll_per_token(
        log_x_theta=log_x_theta, xt=xt, x0=x0, alpha_t=alpha_t,
        dalpha_t=dalpha_t, low_var=low_var)
      self._latest_kl = torch.zeros_like(recon_per_tok)
      self._latest_lb = self._compute_lb_aux(router_logits)
      self._latest_hprior, self._latest_live = self._router_stats(
        router_logits)
      return recon_per_tok

    # Pass A: clean x0 at t = 0 -> posterior router logits q(z | x0).
    t0 = torch.zeros(x0.shape[0], device=self.device, dtype=alpha_t.dtype)
    _, alpha_0 = self.noise(t0)
    sigma_0 = self._sigma_from_alphat(alpha_0.unsqueeze(-1))
    _, clean_router_logits = self._forward_with_routes(x0, sigma_0)

    # Pass B: noisy x_t routed by the posterior.
    raw, noisy_router_logits = self._forward_with_routes(
      xt, sigma_t, force_routes=clean_router_logits)
    log_x_theta = self._process_model_output(raw, xt, sigma_t)
    utils.print_nans(log_x_theta, 'model_output')

    recon_per_tok = self.nll_per_token(
      log_x_theta=log_x_theta, xt=xt, x0=x0,
      alpha_t=alpha_t, dalpha_t=dalpha_t, low_var=low_var)
    self._latest_kl = self._compute_kl(
      clean_router_logits, noisy_router_logits)
    self._latest_lb = (
      self._compute_lb_aux(clean_router_logits)
      + self._compute_lb_aux(noisy_router_logits)) / 2
    self._latest_hprior, self._latest_live = self._router_stats(
      noisy_router_logits)
    return recon_per_tok

  def _loss(self, x0, labels, valid_tokens,
            current_accumulation_step=None, train_mode=False):
    base = super()._loss(
      x0, labels, valid_tokens,
      current_accumulation_step, train_mode)
    if self._latest_kl is None or self._latest_lb is None:
      return base
    vt = valid_tokens.to(self._latest_kl.dtype)
    n_valid = vt.sum(dim=1).clamp_min(1)
    p_mask = self._latest_t.to(self._latest_kl.dtype).clamp_min(1e-3)

    kl_per_sample = (self._latest_kl * vt).sum(dim=1) / n_valid
    kl_term = (kl_per_sample / p_mask).mean()
    lb_term = self._latest_lb
    total = base.loss + self.kl_weight * kl_term + self.lb_weight * lb_term
    if train_mode:
      h_router = ((self._latest_hprior * vt).sum(dim=1) / n_valid).mean()
      for name, value, prog_bar in (
          ('recon_loss', base.loss.detach(), False),
          ('kl_loss', kl_term.detach(), False),
          ('lb_loss', lb_term.detach(), False),
          ('router_entropy', h_router.detach(), True),
          ('live_experts', float(self._latest_live), True)):
        self.log(f'train/{name}', value, on_step=True, on_epoch=False,
                 sync_dist=True, prog_bar=prog_bar, logger=True)
      self.log('train/gumbel_tau', float(self._current_tau()),
               on_step=True, on_epoch=False, sync_dist=False, logger=True)
    self._reset_latest()

    fair_nlls = base.nlls + kl_term.detach() * base.num_tokens
    return trainer_base.Loss(
      loss=total, nlls=fair_nlls,
      prior_loss=0.0, num_tokens=base.num_tokens)


class VADD(trainer_base.AbsorbingState):
  """Variational Absorbing Discrete Diffusion (VADD)."""

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    # Replace the dense DIT backbone with VADD's GenDIT (decoder).
    del self.backbone
    self.backbone = models.dit_vadd.GenDIT(
      self.config,
      vocab_size=self.vocab_size,
      mask_index=self.mask_index)
    # Recognition/encoder network.
    self.encoder = models.dit_vadd.InferDIT(
      self.config, vocab_size=self.vocab_size)
    # Rebuild EMA over (backbone + encoder + noise) parameters.
    if self.ema is not None:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    # VAE-specific config.
    self.latent_dim = int(self.config.algo.latent_dim)
    self.latent1d = bool(self.config.algo.latent1d)
    self.num_particles = int(self.config.algo.num_particles)
    self.init_kl_weight = float(self.config.algo.init_kl_weight)
    self.kl_weight_interval = int(self.config.algo.kl_weight_interval)
    self.loss_strategy = str(self.config.algo.loss_strategy)
    self.multixt = bool(self.config.algo.multixt)
    self.sampling_nz = int(self.config.algo.sampling_nz)
    self.sampling_fixz = bool(self.config.algo.sampling_fixz)
    self._latest_token_nll_w = None
    self._latest_std_dec_mean = None
    self._latest_recon = None
    self._latest_kl = None
    self._latest_kl_weight = None
    self._validate_configuration()

  # ----- Configuration -----
  def _validate_configuration(self):
    assert self.parameterization == 'subs'

  def _get_parameters(self):
    chains = [self.backbone.parameters()]
    if hasattr(self, 'encoder'):
      chains.append(self.encoder.parameters())
    chains.append(self.noise.parameters())
    return itertools.chain(*chains)

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    if hasattr(self, 'encoder'):
      self.encoder.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    if hasattr(self, 'encoder'):
      self.encoder.train()
    self.noise.train()

  # ----- Decoder / encoder forward -----
  def _process_model_output(self, model_output, xt, sigma):
    """MDLM-style subs parameterization on decoder logits."""
    del sigma
    model_output[:, :, self.mask_index] += self.neg_infinity
    model_output = model_output - torch.logsumexp(
      model_output, dim=-1, keepdim=True)
    unmasked_indices = (xt != self.mask_index)
    model_output[unmasked_indices] = self.neg_infinity
    model_output[unmasked_indices, xt[unmasked_indices]] = 0
    return model_output

  def forward(self, xt, sigma, z=None, labels=None, weights=None,
              nn_input_idxs=None):
    """Decoder forward p_theta(x|z, xt). If `z` is None, draws z ~ N(0, I)."""
    del labels, weights, nn_input_idxs
    sigma_proc = self._process_sigma(sigma)
    if z is None:
      z = self._prior_sample_z(xt.shape).to(xt.device).to(self.dtype)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = self.backbone(xt, z, sigma_proc)
    return self._process_model_output(
      model_output=logits, xt=xt, sigma=sigma_proc)

  def infer(self, x0, sigma, xt):
    """Encoder q_phi(z|x0, xt) -> (mean, logstd)."""
    sigma_proc = self._process_sigma(sigma)
    mask_struc = self.config.decoder.mask_struc
    if mask_struc == 'none':
      output = self.encoder(x0, sigma_proc, None)
    else:
      output = self.encoder(x0, sigma_proc, x0 == xt)
    latent_mean, latent_logstd = torch.chunk(output, chunks=2, dim=-1)
    if self.latent1d:
      latent_mean = latent_mean.mean(dim=1)
      latent_logstd = latent_logstd.mean(dim=1)
    return latent_mean, latent_logstd

  # ----- z prior -----
  def _prior_sample_z(self, batch_shape):
    """Sample z ~ N(0, I)."""
    if self.latent1d:
      z = torch.randn(batch_shape[0], self.latent_dim)
    else:
      z = torch.randn(*batch_shape, self.latent_dim)
    return z

  # ----- VAE-ELBO loss -----
  def _vae_loss_components(self, x0, sigma_col, dsigma):
    sigma = sigma_col.squeeze(-1)  # (B,)
    move_chance = (1 - torch.exp(-sigma_col))  # (B, 1)
    unet_conditioning = sigma_col

    if not self.multixt:
      xt = self.q_xt(x0, 1.0 - move_chance)  # AbsorbingState.q_xt uses alpha_t
      latent_mean, latent_logstd = self.infer(x0, unet_conditioning, xt=xt)

    lls, nkls = [], []
    for _ in range(self.num_particles):
      if self.multixt:
        xt = self.q_xt(x0, 1.0 - move_chance)
        latent_mean, latent_logstd = self.infer(
          x0, unet_conditioning, xt=xt)

      eps = torch.randn_like(latent_mean)
      z = latent_logstd.exp() * eps + latent_mean
      logqz = (
        - (z - latent_mean) ** 2 / (latent_logstd * 2.0).exp() / 2.0
        - 0.5 * math.log(2.0 * math.pi)
        - latent_logstd)
      logpz = (
        - z ** 2 / 2.0
        - 0.5 * math.log(2.0 * math.pi))

      model_output = self.forward(xt, unet_conditioning, z=z)
      utils.print_nans(model_output, 'model_output')

      logpxz = torch.gather(
        input=model_output,
        dim=-1,
        index=x0[:, :, None]).squeeze(-1)
      lls.append(logpxz)
      nkls.append(logpz - logqz)

    lls = torch.stack(lls)
    nkls = torch.stack(nkls)
    weight = - (dsigma / torch.expm1(sigma))

    std_of_decoder_mean = torch.mean(torch.std(latent_mean, dim=0))
    return lls, nkls, weight, std_of_decoder_mean

  def _wrapped_vae(self, x0, valid_tokens):
    t = self._sample_t(x0.shape[0], None)
    if self.T > 0:
      t = (t * self.T).to(torch.int) / self.T + (1 / self.T)

    dalpha_t, alpha_t = self.noise(t)
    sigma = self._sigma_from_alphat(alpha_t)
    dsigma = - dalpha_t / alpha_t
    sigma_col = sigma.unsqueeze(-1)

    lls, nkls, weight, std_dec = self._vae_loss_components(
      x0, sigma_col, dsigma)

    # KL warmup
    if self.loss_strategy == 'klweight':
      step = float(self.global_step)
      kl_weight = min(
        step / self.kl_weight_interval
        * (1.0 - self.init_kl_weight) + self.init_kl_weight,
        1.0)
    elif self.loss_strategy == 'none':
      kl_weight = 1.0
    else:
      raise NotImplementedError(self.loss_strategy)

    L = lls.shape[-1]
    K = lls.shape[0]

    if self.latent1d:
      # nkls: (K, B, latent_dim)  lls: (K, B, L)
      recon_term = lls.mean(dim=-1)                                     # (K, B)
      kl_term    = nkls.mean(dim=-1) * (nkls.shape[-1] / L)             # (K, B)
    else:
      # nkls: (K, B, L, latent_dim)
      recon_term = lls.mean(dim=-1)                                     # (K, B)
      kl_term    = nkls.sum(dim=-1).mean(dim=-1)                        # (K, B)

    elbos_w = recon_term + kl_weight * kl_term
    elbos   = recon_term + kl_term

    recon_per_sample = recon_term.mean(dim=0)                           # (B,)
    kl_per_sample = kl_term.mean(dim=0)                                 # (B,)
    self._latest_recon = (recon_per_sample * weight).mean().detach()  
    self._latest_kl = (kl_per_sample * weight).mean().detach()  
    self._latest_kl_weight = float(kl_weight)

    # IWAE bound (per VADD's formula)
    max_elbos_w = torch.max(elbos_w, dim=0, keepdim=True)[0]
    mlb_w = max_elbos_w.squeeze(0) + torch.logsumexp(
      (elbos_w - max_elbos_w) * L - math.log(K), dim=0) / L
    mlb_w = mlb_w * weight 

    max_elbos = torch.max(elbos, dim=0, keepdim=True)[0]
    mlb = max_elbos.squeeze(0) + torch.logsumexp(
      (elbos - max_elbos) * L - math.log(K), dim=0) / L
    mlb = mlb * weight 

    nll_per_token_w = mlb_w.unsqueeze(1).expand(-1, x0.shape[1])
    nll_per_token = mlb.unsqueeze(1).expand(-1, x0.shape[1])
    return nll_per_token_w, nll_per_token, std_dec

  def nll(self, x0, labels, output_tokens,
          current_accumulation_step=None, train_mode=False):
    del labels, output_tokens, current_accumulation_step, train_mode
    nll_w, nll_u, std_dec = self._wrapped_vae(x0, valid_tokens=None)
    self._latest_token_nll_w = nll_w
    self._latest_std_dec_mean = std_dec
    return nll_u

  def _loss(self, x0, labels, valid_tokens,
            current_accumulation_step=None, train_mode=False):
    del labels
    (input_tokens, output_tokens,
     valid_tokens) = self._process_model_input(x0, valid_tokens)
    loss_u = self.nll(input_tokens, None, output_tokens,
                      current_accumulation_step, train_mode)
    assert loss_u.ndim == 2
    if self.ignore_bos:
      valid_tokens[:, 1:] = valid_tokens[:, 1:]

    nlls = (loss_u * valid_tokens).sum()
    num_tokens = valid_tokens.sum()

    nll_w = self._latest_token_nll_w
    weighted_token_nll = (
      (nll_w * valid_tokens).sum() / num_tokens.clamp_min(1))

    if train_mode:
      try:
        if self._latest_std_dec_mean is not None:
          self.log('train/std_dec_mean',
                   self._latest_std_dec_mean.detach(),
                   on_step=True, on_epoch=False, sync_dist=True)
        if self._latest_recon is not None:
          self.log('train/recon_loss', self._latest_recon,
                   on_step=True, on_epoch=False, sync_dist=True)
        if self._latest_kl is not None:
          self.log('train/kl_loss', self._latest_kl,
                   on_step=True, on_epoch=False, sync_dist=True)
        if self._latest_kl_weight is not None:
          self.log('train/kl_weight', float(self._latest_kl_weight),
                   on_step=True, on_epoch=False, sync_dist=False)
      except Exception:
        pass
    # Clear caches.
    self._latest_token_nll_w = None
    self._latest_std_dec_mean = None
    self._latest_recon = None
    self._latest_kl = None
    self._latest_kl_weight = None

    return trainer_base.Loss(
      loss=weighted_token_nll,
      nlls=nlls,
      prior_loss=0.0,
      num_tokens=num_tokens)

  # ----- VAE sampler -----
  def _vae_update(self, x, t, dt, nz=1, z_fix=None):
    """Single VAE diffusion-sampling step (ancestral-style)."""
    _, alpha_t = self.noise(t)
    _, alpha_s = self.noise(t - dt)
    sigma_t = self._sigma_from_alphat(alpha_t).squeeze(-1)
    sigma_s = self._sigma_from_alphat(alpha_s).squeeze(-1)
    if sigma_t.ndim == 0:
      sigma_t = sigma_t.unsqueeze(0)
    if sigma_s.ndim == 0:
      sigma_s = sigma_s.unsqueeze(0)
    move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
    move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
    unet_conditioning = sigma_t.unsqueeze(-1)  # (B, 1)

    q_xs_tensor = []
    for _ in range(nz):
      if z_fix is None:
        z = self._prior_sample_z(x.shape).to(x.device).to(self.dtype)
      else:
        z = z_fix
      log_p_x0 = self.forward(x, unet_conditioning, z=z)
      assert move_chance_t.ndim == log_p_x0.ndim
      q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      q_xs_tensor.append(q_xs)
    q_xs = torch.mean(torch.stack(q_xs_tensor), dim=0)
    _x = trainer_base.sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  @torch.no_grad()
  def generate_samples(self, num_samples, labels=None,
                       num_steps=None, eps=1e-5):
    del labels
    if num_steps is None:
      num_steps = self.config.sampling.steps

    x = self.prior_sample(num_samples, self.num_tokens)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1.0 - eps) / num_steps

    if self.sampling_fixz:
      z_fix = self._prior_sample_z(x.shape).to(x.device).to(self.dtype)
    else:
      z_fix = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
      x = self._vae_update(x, t, dt, nz=self.sampling_nz, z_fix=z_fix)

    if self.config.sampling.noise_removal == 'greedy':
      t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
      _, alpha_t = self.noise(t)
      sigma = self._sigma_from_alphat(alpha_t)
      z = (self._prior_sample_z(x.shape).to(x.device).to(self.dtype)
           if z_fix is None else z_fix)
      x = self.forward(x, sigma, z=z).argmax(dim=-1)
    elif self.config.sampling.noise_removal == 'ancestral':
      t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
      _, alpha_t = self.noise(t)
      sigma = self._sigma_from_alphat(alpha_t)
      z = (self._prior_sample_z(x.shape).to(x.device).to(self.dtype)
           if z_fix is None else z_fix)
      # Greedy decode as final step.
      x = self.forward(x, sigma, z=z).argmax(dim=-1)
    return x
