"""
DF-SSM Mamba-2 Distillation — Density Field Weights (DFW)
==========================================================
Like df_ssm_mamba2_distill.py but with DENSITY FIELD WEIGHTS:
  - Weights quantized to K_w²+1 levels via spray matrices (not binary sign)
  - Smoother loss landscape → faster convergence → fewer tokens needed
  - Deploys as binary (same popcount hardware)

Key difference from standard binary distillation:
  Standard:  weight = sign(w_latent) × scale               → 2 values
  DFW K=4:   weight = spray_quantize(sigmoid(w_latent)) × scale → 17 values
  DFW K=8:   weight = spray_quantize(sigmoid(w_latent)) × scale → 65 values

  Training sees 17-65 levels (smooth gradients, fast convergence)
  Inference uses sign(w_latent) or density field readout (pure binary)

Usage:
  python df_ssm_mamba2_distill_dfw.py --quick --Kw 4              # 10M tokens
  python df_ssm_mamba2_distill_dfw.py --tokens 100M --Kw 4        # should converge much faster
  python df_ssm_mamba2_distill_dfw.py --tokens 100M --Kw 8        # even smoother

Hardware: same as standard distillation (A100 40GB)
"""

import argparse
import math
import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import warnings
warnings.filterwarnings('ignore', message='.*lr_scheduler.step.*optimizer.step.*')


# ============================================================
# SSD Core (same as standard distillation)
# ============================================================

def _segsum(x):
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_scan_no_df(X, A, B, C, block_len):
    """Standard SSD — no density field. Used by teacher."""
    assert X.shape[1] % block_len == 0
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)
    L = torch.exp(_segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, 1:]
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
    return rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")


# ============================================================
# Spray Bank (shared by state DF and weight DF)
# ============================================================

class SprayBank(nn.Module):
    """Pre-computed sorted thresholds for density field quantization."""
    def __init__(self, K=8, num_banks=256):
        super().__init__()
        self.K = K
        self.KK = K * K
        thresholds = torch.rand(num_banks, self.KK).sort(dim=-1).values
        self.register_buffer('thresholds', thresholds)

    def quantize(self, h_scaled, bank_idx=0):
        R = self.thresholds[bank_idx % len(self.thresholds)]
        count = torch.searchsorted(R.to(h_scaled.device), h_scaled.contiguous())
        return count.float() / self.KK


# ============================================================
# State Density Field (same as standard distillation)
# ============================================================

class DensityFieldConfig:
    def __init__(self, K=8, use_sigma_delta=True, sd_accumulator_bits=8,
                 block_len=64):
        self.K = K
        self.use_sigma_delta = use_sigma_delta
        self.sd_accumulator_bits = sd_accumulator_bits
        self.block_len = block_len


def _quantize_error(e, bits=8):
    if bits >= 32:
        return e
    levels = 2 ** bits
    return torch.round(e.clamp(-1, 1) * (levels / 2)) / (levels / 2)


def df_ssd_scan(X, A_disc, B_heads, C_heads, block_len, spray_bank, df_config):
    """SSD with density field quantization at chunk boundaries."""
    B_sz, L, nheads, headdim = X.shape
    d_state = B_heads.shape[-1]

    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len))
        A_disc = F.pad(A_disc, (0, 0, 0, pad_len))
        B_heads = F.pad(B_heads, (0, 0, 0, 0, 0, pad_len))
        C_heads = F.pad(C_heads, (0, 0, 0, 0, 0, pad_len))

    num_chunks = X.shape[1] // block_len
    Xc = rearrange(X, "b (c l) h p -> b c l h p", l=block_len)
    Ac = rearrange(A_disc, "b (c l) h -> b h c l", l=block_len)
    Bc = rearrange(B_heads, "b (c l) h n -> b c l h n", l=block_len)
    Cc = rearrange(C_heads, "b (c l) h n -> b c l h n", l=block_len)
    A_cumsum = torch.cumsum(Ac, dim=-1)

    L_mask = torch.exp(_segsum(Ac))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", Cc, Bc, L_mask, Xc)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    intra_states = torch.einsum("bclhn,bhcl,bclhp->bchpn", Bc, decay_states, Xc)
    chunk_decay = torch.exp(A_cumsum[:, :, :, -1])

    running_h = torch.zeros(B_sz, nheads, headdim, d_state,
                            device=X.device, dtype=X.dtype)
    sd_error = torch.zeros_like(running_h) if df_config.use_sigma_delta else None
    start_states = []

    for c in range(num_chunks):
        start_states.append(running_h)
        decay_c = chunk_decay[:, :, c].unsqueeze(-1).unsqueeze(-1)
        running_h = decay_c * running_h + intra_states[:, c]

        h_mean = running_h.mean(dim=-1, keepdim=True)
        h_std = running_h.std(dim=-1, keepdim=True).clamp(min=1e-8)
        h_lo = h_mean - 3.0 * h_std
        h_range = (6.0 * h_std).clamp(min=1e-8)
        h_scaled = ((running_h - h_lo) / h_range).clamp(0, 1)

        if df_config.use_sigma_delta and sd_error is not None:
            h_corr = (h_scaled + sd_error).clamp(0, 1)
        else:
            h_corr = h_scaled

        h_quantized = spray_bank.quantize(h_corr, bank_idx=c)

        if df_config.use_sigma_delta:
            sd_error = _quantize_error(h_corr - h_quantized,
                                       df_config.sd_accumulator_bits)

        running_h = h_quantized * h_range + h_lo

    start_states = torch.stack(start_states, dim=1)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", Cc, start_states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y[:, :L, :, :]


# ============================================================
# Density Field Linear — THE KEY INNOVATION
# ============================================================

class STEClamp(torch.autograd.Function):
    """Clamp to [-1, 1] in forward, pass gradient through in backward (STE)."""
    @staticmethod
    def forward(ctx, x):
        return x.clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # gradient passes through unchanged


class DFWQuantize(torch.autograd.Function):
    """Quantize via spray bank with straight-through estimator.
    
    Forward: target → spray_quantize → K²+1 discrete levels
    Backward: gradient passes through unchanged (STE)
    """
    @staticmethod
    def forward(ctx, target_density, spray_bank, step):
        quantized = spray_bank.quantize(target_density, bank_idx=step)
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class DensityFieldLinear(nn.Module):
    """Linear layer with density field weight quantization.
    
    Instead of sign(w) × scale (2 values), uses:
      density = spray_quantize(sigmoid(w_latent))   → K²+1 levels in [0,1]
      effective_w = (density * 2 - 1) × scale       → K²+1 levels in [-scale, +scale]
    
    K=4: 17 levels   (like 4-bit training, deploys as binary)
    K=8: 65 levels   (like 6-bit training, deploys as binary)
    
    At inference/export: sign(w_latent) gives binary weights for hardware.
    """
    def __init__(self, in_features, out_features, bias=False, Kw=4, anneal_steps=2000):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Kw = Kw
        self.latent_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.scale = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.spray_bank = SprayBank(K=Kw, num_banks=64)
        self._step = 0
        self.anneal_steps = anneal_steps  # steps to go from continuous → fully quantized

    def forward(self, x):
        # Map latent weights to [0,1] density targets
        # Linear clamp: gradient = 1.0 everywhere (STE), no saturation
        target_density = (STEClamp.apply(self.latent_weight) + 1) / 2

        # Quantize to K²+1 levels via spray matrix (STE for gradients)
        quantized = DFWQuantize.apply(target_density, self.spray_bank, self._step)

        # Anneal: blend continuous → quantized
        # Step 0:    alpha=0, density = target_density (continuous, zero init error)
        # Step N:    alpha=1, density = quantized (fully discrete, deployment-ready)
        alpha = min(self._step / max(self.anneal_steps, 1), 1.0)
        density = (1 - alpha) * target_density + alpha * quantized

        self._step += 1

        # Map density [0,1] back to effective weight [-1, +1]
        effective_w = density * 2.0 - 1.0

        # Apply per-channel scale
        weight = effective_w * self.scale.unsqueeze(1)
        return F.linear(x, weight, self.bias)

    @torch.no_grad()
    def init_from_fp(self, fp_linear):
        """Initialize from full-precision weights.
        
        Normalizes weights to [-1, 1] directly. No sigmoid, no inverse.
        """
        w = fp_linear.weight.data.float()
        self.scale.data = w.abs().mean(dim=1)

        # Normalize to [-1, 1] — this IS the latent weight (no transform needed)
        w_norm = w / (self.scale.data.unsqueeze(1) + 1e-8)
        self.latent_weight.data = w_norm.clamp(-1, 1)

        if self.bias is not None and fp_linear.bias is not None:
            self.bias.data = fp_linear.bias.data.float()

    @torch.no_grad()
    def export_binary(self):
        """Export as deterministic binary weights for hardware deployment."""
        return self.latent_weight.sign().to(torch.int8)

    @torch.no_grad()
    def weight_stats(self):
        """Report weight distribution statistics."""
        p = (self.latent_weight.clamp(-1, 1) + 1) / 2  # [0,1]
        confident = ((p > 0.9) | (p < 0.1)).float().mean()
        uncertain = ((p > 0.4) & (p < 0.6)).float().mean()
        return {
            'confident_pct': confident.item() * 100,
            'uncertain_pct': uncertain.item() * 100,
            'mean_abs_latent': self.latent_weight.abs().mean().item(),
        }


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


# ============================================================
# Student Model (density field weights + DF-SSD state)
# ============================================================

class DFWMamba2Block(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2,
                 headdim=64, ngroups=1, Kw=4, anneal_steps=2000):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups

        self.d_xBC = self.d_inner + 2 * ngroups * d_state
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads

        # Density field linear layers (K²+1 levels instead of 2)
        self.in_proj = DensityFieldLinear(d_model, d_in_proj, bias=False, Kw=Kw, anneal_steps=anneal_steps)
        self.out_proj = DensityFieldLinear(self.d_inner, d_model, bias=False, Kw=Kw, anneal_steps=anneal_steps)

        self.conv1d = nn.Conv1d(self.d_xBC, self.d_xBC, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_xBC, bias=True)
        self.act = nn.SiLU()

        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.norm = RMSNorm(self.d_inner)

    def forward(self, x, df_config=None, spray_bank=None):
        B, L, D = x.shape

        proj = self.in_proj(x)
        z = proj[:, :, :self.d_inner]
        xBC = proj[:, :, self.d_inner:self.d_inner + self.d_xBC]
        dt_raw = proj[:, :, self.d_inner + self.d_xBC:]

        xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :L]).transpose(1, 2)

        x_ssm = xBC[:, :, :self.d_inner]
        B_param = xBC[:, :, self.d_inner:self.d_inner + self.ngroups * self.d_state]
        C_param = xBC[:, :, self.d_inner + self.ngroups * self.d_state:]

        dt = F.softplus(dt_raw + self.dt_bias)
        A = -torch.exp(self.A_log.float())

        x_heads = x_ssm.float().reshape(B, L, self.nheads, self.headdim)
        hpg = self.nheads // self.ngroups

        X = x_heads * dt.float().unsqueeze(-1)
        A_disc = dt.float() * A.unsqueeze(0)
        B_heads = B_param.float().reshape(B, L, self.ngroups, self.d_state)
        B_heads = B_heads.repeat_interleave(hpg, dim=2)
        C_heads = C_param.float().reshape(B, L, self.ngroups, self.d_state)
        C_heads = C_heads.repeat_interleave(hpg, dim=2)

        block_len = df_config.block_len if df_config else 64
        if df_config is not None and spray_bank is not None:
            Y = df_ssd_scan(X, A_disc, B_heads, C_heads, block_len,
                            spray_bank, df_config)
        else:
            pad_len = (block_len - L % block_len) % block_len
            if pad_len > 0:
                X_p = F.pad(X, (0, 0, 0, 0, 0, pad_len))
                A_p = F.pad(A_disc, (0, 0, 0, pad_len))
                B_p = F.pad(B_heads, (0, 0, 0, 0, 0, pad_len))
                C_p = F.pad(C_heads, (0, 0, 0, 0, 0, pad_len))
            else:
                X_p, A_p, B_p, C_p = X, A_disc, B_heads, C_heads
            Y = ssd_scan_no_df(X_p, A_p, B_p, C_p, block_len)[:, :L, :, :]

        Y = Y + x_heads * self.D.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        y = Y.reshape(B, L, self.d_inner)
        y = self.norm(y * F.silu(z.float()))
        return self.out_proj(y)


class DFWMamba2ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2,
                 headdim=64, ngroups=1, Kw=4, anneal_steps=2000):
        super().__init__()
        self.mixer = DFWMamba2Block(d_model, d_state, d_conv, expand,
                                    headdim, ngroups, Kw, anneal_steps)
        self.norm = RMSNorm(d_model)

    def forward(self, x, df_config=None, spray_bank=None):
        return x + self.mixer(self.norm(x), df_config=df_config, spray_bank=spray_bank)


class DFWMamba2LM(nn.Module):
    def __init__(self, d_model, n_layer, vocab_size, d_state=128,
                 d_conv=4, expand=2, headdim=64, ngroups=1, Ks=8, Kw=4, anneal_steps=2000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DFWMamba2ResidualBlock(d_model, d_state, d_conv, expand,
                                   headdim, ngroups, Kw, anneal_steps)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.spray_bank = SprayBank(K=Ks, num_banks=256)

    def forward(self, input_ids, df_config=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, df_config, self.spray_bank,
                    use_reentrant=False,
                )
            else:
                x = layer(x, df_config=df_config, spray_bank=self.spray_bank)
        x = self.norm_f(x)
        return self.lm_head(x)

    def weight_stats(self):
        """Report weight distribution across all DFW layers."""
        stats = []
        for m in self.modules():
            if isinstance(m, DensityFieldLinear):
                stats.append(m.weight_stats())
        if not stats:
            return {}
        return {
            'confident_pct': np.mean([s['confident_pct'] for s in stats]),
            'uncertain_pct': np.mean([s['uncertain_pct'] for s in stats]),
            'mean_abs_latent': np.mean([s['mean_abs_latent'] for s in stats]),
        }


# ============================================================
# Teacher Model (same as standard distillation)
# ============================================================

class TeacherMamba2Block(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2,
                 headdim=64, ngroups=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups

        self.d_xBC = self.d_inner + 2 * ngroups * d_state
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(self.d_xBC, self.d_xBC, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_xBC, bias=True)
        self.act = nn.SiLU()

        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        proj = self.in_proj(x)
        z = proj[:, :, :self.d_inner]
        xBC = proj[:, :, self.d_inner:self.d_inner + self.d_xBC]
        dt_raw = proj[:, :, self.d_inner + self.d_xBC:]
        xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :L]).transpose(1, 2)
        x_ssm = xBC[:, :, :self.d_inner]
        B_param = xBC[:, :, self.d_inner:self.d_inner + self.ngroups * self.d_state]
        C_param = xBC[:, :, self.d_inner + self.ngroups * self.d_state:]
        dt = F.softplus(dt_raw + self.dt_bias)
        A = -torch.exp(self.A_log.float())
        x_heads = x_ssm.float().reshape(B, L, self.nheads, self.headdim)
        hpg = self.nheads // self.ngroups
        X = x_heads * dt.float().unsqueeze(-1)
        A_disc = dt.float() * A.unsqueeze(0)
        B_heads = B_param.float().reshape(B, L, self.ngroups, self.d_state)
        B_heads = B_heads.repeat_interleave(hpg, dim=2)
        C_heads = C_param.float().reshape(B, L, self.ngroups, self.d_state)
        C_heads = C_heads.repeat_interleave(hpg, dim=2)
        block_len = 64
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            X = F.pad(X, (0, 0, 0, 0, 0, pad_len))
            A_disc = F.pad(A_disc, (0, 0, 0, pad_len))
            B_heads = F.pad(B_heads, (0, 0, 0, 0, 0, pad_len))
            C_heads = F.pad(C_heads, (0, 0, 0, 0, 0, pad_len))
        Y = ssd_scan_no_df(X, A_disc, B_heads, C_heads, block_len)
        Y = Y[:, :L, :, :]
        Y = Y + x_heads[:, :L] * self.D.float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        y = Y.reshape(B, L, self.d_inner)
        y = self.norm(y * F.silu(z.float()))
        return self.out_proj(y.to(self.out_proj.weight.dtype))


class TeacherMamba2ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2,
                 headdim=64, ngroups=1):
        super().__init__()
        self.mixer = TeacherMamba2Block(d_model, d_state, d_conv, expand,
                                        headdim, ngroups)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class TeacherMamba2LM(nn.Module):
    def __init__(self, d_model, n_layer, vocab_size, d_state=128,
                 d_conv=4, expand=2, headdim=64, ngroups=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TeacherMamba2ResidualBlock(d_model, d_state, d_conv, expand,
                                       headdim, ngroups)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x.to(self.lm_head.weight.dtype))


# ============================================================
# Weight Loading
# ============================================================

def load_teacher(model_name, device='cpu'):
    from huggingface_hub import hf_hub_download
    import json as json_module

    print(f"  Loading teacher: {model_name}", flush=True)
    try:
        config_path = hf_hub_download(model_name, 'config.json')
        with open(config_path) as f:
            raw = json_module.load(f)
        ssm = raw.get('ssm_cfg', {})
        cfg = dict(d_model=raw.get('d_model', 2048), n_layer=raw.get('n_layer', 64),
                   d_state=ssm.get('d_state', 128), d_conv=ssm.get('d_conv', 4),
                   expand=ssm.get('expand', 2), headdim=ssm.get('headdim', 64),
                   ngroups=ssm.get('ngroups', 1))
    except:
        cfg = dict(d_model=2048, n_layer=64, d_state=128, d_conv=4,
                   expand=2, headdim=64, ngroups=1)

    try:
        weights_path = hf_hub_download(model_name, 'pytorch_model.bin')
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    except:
        from safetensors.torch import load_file
        weights_path = hf_hub_download(model_name, 'model.safetensors')
        state_dict = load_file(weights_path)

    for k, v in state_dict.items():
        if 'embedding' in k and 'weight' in k:
            vocab_size = v.shape[0]; break
    else:
        vocab_size = 50288

    d_inner = cfg['d_model'] * cfg['expand']
    nheads = d_inner // cfg['headdim']
    for k, v in state_dict.items():
        if 'layers.0.mixer.in_proj.weight' in k:
            remainder = v.shape[0] - 2 * d_inner - nheads
            for ds in [128, 64, 256]:
                for ng in [1, 2, 4, 8]:
                    if 2 * ng * ds == remainder:
                        cfg['d_state'] = ds; cfg['ngroups'] = ng; break
                else: continue
                break
            break

    print(f"  Config: d_model={cfg['d_model']}, n_layer={cfg['n_layer']}, "
          f"d_state={cfg['d_state']}, nheads={nheads}, ngroups={cfg['ngroups']}", flush=True)

    teacher = TeacherMamba2LM(vocab_size=vocab_size, **cfg)
    mapped = {}
    for k, v in state_dict.items():
        new_k = k.replace('backbone.embedding', 'embedding')
        new_k = new_k.replace('backbone.layers.', 'layers.')
        new_k = new_k.replace('backbone.norm_f', 'norm_f')
        mapped[new_k] = v

    missing, unexpected = teacher.load_state_dict(mapped, strict=False)
    if missing:
        real = [k for k in missing if 'bias' not in k or 'dt_bias' in k or 'conv1d' in k]
        if real: print(f"  Warning — missing: {real[:5]}")

    teacher = teacher.to(device).half()
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params, FP16", flush=True)
    return teacher, cfg, vocab_size


def init_student_from_teacher(student, teacher):
    print("  Initializing student from teacher...", flush=True)
    student.embedding.weight.data = teacher.embedding.weight.data.float()
    student.norm_f.weight.data = teacher.norm_f.weight.data.float()

    for s_layer, t_layer in zip(student.layers, teacher.layers):
        s_layer.norm.weight.data = t_layer.norm.weight.data.float()
        s, t = s_layer.mixer, t_layer.mixer
        s.in_proj.init_from_fp(t.in_proj)
        s.out_proj.init_from_fp(t.out_proj)
        s.conv1d.weight.data = t.conv1d.weight.data.float()
        s.conv1d.bias.data = t.conv1d.bias.data.float()
        s.A_log.data = t.A_log.data.float()
        s.D.data = t.D.data.float()
        s.dt_bias.data = t.dt_bias.data.float()
        s.norm.weight.data = t.norm.weight.data.float()

    dfw_params = sum(m.latent_weight.numel() for m in student.modules()
                     if isinstance(m, DensityFieldLinear))
    total_params = sum(p.numel() for p in student.parameters())
    ws = student.weight_stats()
    print(f"  Student: {total_params/1e6:.1f}M total ({dfw_params/1e6:.1f}M DFW, "
          f"{(total_params-dfw_params)/1e6:.1f}M FP)", flush=True)
    print(f"  Weight stats: {ws['confident_pct']:.1f}% confident, "
          f"{ws['uncertain_pct']:.1f}% uncertain, "
          f"mean|latent|={ws['mean_abs_latent']:.3f}", flush=True)


# ============================================================
# Data Loading
# ============================================================

def get_data_iterator(tokenizer, seq_len=512, batch_size=1, device='cpu'):
    from datasets import load_dataset
    print(f"  Loading C4 dataset (streaming, batch_size={batch_size})...", flush=True)
    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    buffer = []
    for item in dataset:
        tokens = tokenizer(item['text'], truncation=False, add_special_tokens=False)['input_ids']
        buffer.extend(tokens)
        while len(buffer) >= batch_size * (seq_len + 1):
            batch_inputs, batch_targets = [], []
            for _ in range(batch_size):
                chunk = buffer[:seq_len + 1]
                buffer = buffer[seq_len:]
                batch_inputs.append(chunk[:-1])
                batch_targets.append(chunk[1:])
            yield (torch.tensor(batch_inputs, dtype=torch.long, device=device),
                   torch.tensor(batch_targets, dtype=torch.long, device=device))


# ============================================================
# Training
# ============================================================

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up dual logging: screen + file
    import sys
    log_file = args.log_file or f'dfw_train_{args.Kw}_{int(time.time())}.log'

    class TeeLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'a')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = TeeLogger(log_file)
    print(f"  Logging to: {log_file}")

    print("=" * 70)
    print("DF-SSM MAMBA-2 DISTILLATION — DENSITY FIELD WEIGHTS (DFW)")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Device:      {device}")
    print(f"  Tokens:      {args.tokens:,}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seq length:  {args.seq_len}")
    print(f"  Block len:   {args.block_len}")
    print(f"  Ks (state):  {args.Ks}")
    print(f"  Kw (weight): {args.Kw}  → {args.Kw**2 + 1} weight levels")
    print(f"  Anneal:      {args.anneal_steps} steps (continuous → quantized)")
    print(f"  Grad accum:  {args.grad_accum}")
    print(f"  LR:          {args.lr}")
    print(f"  LR DFW mult: {args.lr_dfw_mult}× ({args.lr * args.lr_dfw_mult:.1e} for DFW weights)")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha (KL):  {args.alpha}")
    print(f"  Optimizer:   {args.optimizer}")
    print()

    teacher, cfg, vocab_size = load_teacher(args.model, device=device)
    student = DFWMamba2LM(vocab_size=vocab_size, Ks=args.Ks, Kw=args.Kw,
                          anneal_steps=args.anneal_steps, **cfg).to(device)
    init_student_from_teacher(student, teacher)

    df_config = DensityFieldConfig(K=args.Ks, use_sigma_delta=True,
                                    sd_accumulator_bits=8, block_len=args.block_len)

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except:
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

    # Optimizer — separate LR for DFW latent weights (need higher LR due to sigmoid saturation)
    dfw_params = [(n, p) for n, p in student.named_parameters() if 'latent_weight' in n]
    other_params = [(n, p) for n, p in student.named_parameters() if 'latent_weight' not in n]
    dfw_lr = args.lr * args.lr_dfw_mult
    param_groups = [
        {'params': [p for _, p in dfw_params], 'lr': dfw_lr},
        {'params': [p for _, p in other_params], 'lr': args.lr},
    ]
    print(f"  DFW weight LR: {dfw_lr:.1e} ({args.lr_dfw_mult}× base)", flush=True)
    print(f"  Other params LR: {args.lr:.1e}", flush=True)

    use_schedulefree = (args.optimizer == 'schedulefree')
    if use_schedulefree:
        try:
            import schedulefree
            optimizer = schedulefree.AdamWScheduleFree(
                param_groups,
                weight_decay=0.01, betas=(0.9, 0.95),
                warmup_steps=min(1000, args.tokens // (args.seq_len * args.batch_size * args.grad_accum) // 10),
            )
            print("  Using Schedule-Free AdamW", flush=True)
        except ImportError:
            print("  schedulefree not installed, falling back to AdamW", flush=True)
            use_schedulefree = False

    if not use_schedulefree:
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(param_groups,
                                                 weight_decay=0.01, betas=(0.9, 0.95))
                print("  Using 8-bit AdamW", flush=True)
            except ImportError:
                optimizer = torch.optim.AdamW(param_groups,
                                               weight_decay=0.01, betas=(0.9, 0.95))
                print("  8-bit Adam unavailable, using standard AdamW", flush=True)
        else:
            optimizer = torch.optim.AdamW(param_groups,
                                           weight_decay=0.01, betas=(0.9, 0.95))
            print("  Using standard AdamW", flush=True)

    total_steps = args.tokens // (args.seq_len * args.batch_size * args.grad_accum)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else min(1000, total_steps // 10)
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}", flush=True)

    if use_schedulefree:
        scheduler = None
    else:
        def lr_schedule(step):
            if step < warmup_steps: return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    data_iter = get_data_iterator(tokenizer, seq_len=args.seq_len,
                                    batch_size=args.batch_size, device=device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU memory after load: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    print(f"\n{'='*70}\nTRAINING\n{'='*70}")

    # CSV log for graphing
    csv_file = log_file.replace('.log', '.csv')
    write_csv_header = not os.path.exists(csv_file) or not args.resume
    csv_f = open(csv_file, 'a')
    if write_csv_header:
        csv_f.write('step,tokens_M,loss,lm_loss,kl_loss,ppl,lr_base,lr_dfw,quant_alpha,confident_pct,latent_abs\n')
        csv_f.flush()
    print(f"  CSV log: {csv_file}")

    student.train()
    total_tokens = 0
    step = 0
    accum_loss = accum_lm = accum_kl = 0
    start_time = time.time()
    optimizer.zero_grad()

    if use_schedulefree:
        optimizer.train()

    use_amp = (device == 'cuda' and not args.no_amp)
    auto_amp = use_amp  # if AMP requested, start disabled and auto-enable
    if auto_amp:
        use_amp = False  # start with FP32
        amp_loss_threshold = 5000  # enable AMP when loss drops below this
        print("  Auto-AMP: starting FP32, will enable AMP when loss < 5000", flush=True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if not use_amp and not auto_amp:
        print("  AMP disabled (FP32 training)", flush=True)

    # Resume
    if args.resume:
        print(f"  Resuming from {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = student.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing:
            real_missing = [k for k in missing if 'spray_bank' not in k]
            if real_missing:
                print(f"  WARNING: missing keys: {real_missing[:5]}", flush=True)
            spray_missing = [k for k in missing if 'spray_bank' in k]
            if spray_missing:
                print(f"  SprayBank reinitialized ({len(spray_missing)} buffers, K may have changed)", flush=True)
        if args.reset_optimizer:
            print(f"  Optimizer RESET (fresh schedule-free state, weights kept)", flush=True)
            step = 0
            total_tokens = 0
        else:
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
                try:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                except RuntimeError:
                    print("  Scaler state incompatible, starting fresh", flush=True)
            step = ckpt.get('step', 0)
            total_tokens = ckpt.get('total_tokens', 0)
            # Override LR from CLI (checkpoint may have different LR)
            optimizer.param_groups[0]['lr'] = dfw_lr
            optimizer.param_groups[1]['lr'] = args.lr
            print(f"  LR overridden: DFW={dfw_lr:.1e}, other={args.lr:.1e}", flush=True)
        print(f"  Resumed at step {step}, {total_tokens/1e6:.1f}M tokens", flush=True)
        skip_batches = total_tokens // (args.seq_len * args.batch_size)
    else:
        skip_batches = 0

    # Ctrl+C handler
    interrupted = [False]
    import signal
    def handle_interrupt(signum, frame):
        if interrupted[0]:
            raise SystemExit(1)
        interrupted[0] = True
        print(f"\n  Ctrl+C caught. Saving checkpoint...", flush=True)
    signal.signal(signal.SIGINT, handle_interrupt)

    for batch_idx, (input_ids, targets) in enumerate(data_iter):
        if batch_idx < skip_batches:
            if batch_idx % 1000 == 0 and batch_idx > 0:
                print(f"    skipping... {batch_idx}/{skip_batches}", flush=True)
            continue

        if interrupted[0]:
            save_checkpoint(student, optimizer, scheduler, scaler, step, total_tokens, args)
            csv_f.close()
            print(f"  Resume with: --resume dfssm_dfw_step{step}.pt", flush=True)
            return

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                teacher_logits = teacher(input_ids).float()

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            student_logits = student(input_ids, df_config=df_config)
            T = args.temperature
            kl_loss = F.kl_div(F.log_softmax(student_logits.float() / T, dim=-1),
                               F.softmax(teacher_logits / T, dim=-1),
                               reduction='batchmean') * (T * T)
            lm_loss = F.cross_entropy(student_logits.float().reshape(-1, student_logits.size(-1)),
                                       targets.reshape(-1))
            loss = args.alpha * kl_loss + (1 - args.alpha) * lm_loss

        # NaN protection — check before backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: NaN/Inf loss at batch {batch_idx}, skipping", flush=True)
            if auto_amp and use_amp:
                use_amp = False
                scaler = torch.amp.GradScaler('cuda', enabled=False)
                print(f"    >>> AMP DISABLED (NaN detected, falling back to FP32)", flush=True)
            optimizer.zero_grad()
            continue

        scaler.scale(loss / args.grad_accum).backward()

        accum_loss += loss.item()
        accum_lm += lm_loss.item()
        accum_kl += kl_loss.item()
        total_tokens += input_ids.numel()

        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"    batch {batch_idx:>5d} | loss={loss.item():.2f} | "
                  f"{total_tokens/max(elapsed,1e-6):.1f} tok/s | "
                  f"{total_tokens/1e6:.2f}M tok", flush=True)

        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 5 == 0:
                avg_loss = accum_loss / (5 * args.grad_accum)
                avg_lm = accum_lm / (5 * args.grad_accum)
                avg_kl = accum_kl / (5 * args.grad_accum)
                elapsed = time.time() - start_time
                ppl = math.exp(min(avg_lm, 20))
                lr_dfw = optimizer.param_groups[0]['lr']
                lr_other = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else lr_dfw

                # Auto-AMP: enable when loss is safely within FP16 range
                if auto_amp and not use_amp and avg_loss < amp_loss_threshold:
                    use_amp = True
                    scaler = torch.amp.GradScaler('cuda', enabled=True)
                    print(f"    >>> AMP ENABLED (loss={avg_loss:.0f} < {amp_loss_threshold})", flush=True)
                # Auto-AMP: disable if loss spikes back above threshold (e.g., alpha=1.0 transition)
                elif auto_amp and use_amp and avg_loss > amp_loss_threshold * 1.5:
                    use_amp = False
                    scaler = torch.amp.GradScaler('cuda', enabled=False)
                    print(f"    >>> AMP DISABLED (loss={avg_loss:.0f} > {amp_loss_threshold * 1.5:.0f})", flush=True)

                # Get alpha
                for m in student.modules():
                    if isinstance(m, DensityFieldLinear):
                        alpha = min(m._step / max(m.anneal_steps, 1), 1.0)
                        break
                else:
                    alpha = 0.0
                amp_str = "AMP" if use_amp else "FP32"
                mem = f" | {torch.cuda.max_memory_allocated()/1e9:.1f}GB" if device == 'cuda' else ""
                print(f"  Step {step:>6d} | Loss {avg_loss:.4f} (LM {avg_lm:.4f} KL {avg_kl:.4f}) | "
                      f"PPL {ppl:.1f} | LR {lr_other:.2e}/{lr_dfw:.2e} | α={alpha:.2f} | {amp_str} | "
                      f"{total_tokens/elapsed:.0f} tok/s | {total_tokens/1e6:.1f}M{mem}", flush=True)
                accum_loss = accum_lm = accum_kl = 0

                # CSV log every 5 steps
                ws = student.weight_stats()
                csv_f.write(f'{step},{total_tokens/1e6:.2f},{avg_loss:.4f},{avg_lm:.4f},'
                            f'{avg_kl:.4f},{ppl:.1f},{lr_other:.6f},{lr_dfw:.6f},'
                            f'{alpha:.4f},{ws["confident_pct"]:.1f},{ws["mean_abs_latent"]:.4f}\n')
                csv_f.flush()

            # Weight stats every 100 steps (screen only, CSV already has it)
            if step % 100 == 0:
                ws = student.weight_stats()
                for m in student.modules():
                    if isinstance(m, DensityFieldLinear):
                        alpha = min(m._step / max(m.anneal_steps, 1), 1.0)
                        break
                else:
                    alpha = 0.0
                print(f"    Weights: {ws['confident_pct']:.1f}% confident, "
                      f"{ws['uncertain_pct']:.1f}% uncertain, "
                      f"|latent|={ws['mean_abs_latent']:.3f}, "
                      f"quant_alpha={alpha:.2f}", flush=True)

            if step % 2000 == 0:
                save_checkpoint(student, optimizer, scheduler, scaler, step, total_tokens, args)

        if total_tokens >= args.tokens: break

    elapsed = time.time() - start_time
    print(f"\n  Done: {total_tokens/1e6:.1f}M tokens, {step} steps, {elapsed/3600:.1f}h", flush=True)
    csv_f.close()

    if use_schedulefree:
        optimizer.eval()
    save_checkpoint(student, optimizer, scheduler, scaler, step, total_tokens, args, final=True)
    evaluate_student(student, teacher, tokenizer, df_config, device, args)


def save_checkpoint(student, optimizer, scheduler, scaler, step, total_tokens, args, final=False):
    suffix = 'final' if final else f'step{step}'
    path = f'dfssm_dfw_{suffix}.pt'
    save_dict = {'step': step, 'total_tokens': total_tokens,
                'args': vars(args),
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()}
    if scheduler:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, path)
    print(f"  Saved: {path}", flush=True)

    if final:
        # Export binary weights
        export = {}
        for name, module in student.named_modules():
            if isinstance(module, DensityFieldLinear):
                export[name + '.binary_weight'] = module.export_binary()
                export[name + '.scale'] = module.scale.data.half()
                if module.bias is not None:
                    export[name + '.bias'] = module.bias.data.half()

        # Copy non-DFW parameters
        for name, param in student.state_dict().items():
            if 'latent_weight' not in name and 'spray_bank' not in name:
                already = any(name.startswith(k) for k in export)
                if not already:
                    export[name] = param.half()

        torch.save(export, 'dfssm_dfw_binary.pt')
        size_bits = sum(v.numel() * (1 if v.dtype == torch.int8 else 16) for v in export.values())
        print(f"  Binary model: dfssm_dfw_binary.pt ({size_bits/8/1024/1024:.0f} MB)", flush=True)


@torch.no_grad()
def evaluate_student(student, teacher, tokenizer, df_config, device, args):
    student.eval()
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        texts = [item['text'] for item in ds if item['text'].strip()]
    except:
        texts = ["The quick brown fox. " * 500]

    input_ids = tokenizer('\n'.join(texts), return_tensors='pt', truncation=False).input_ids[0]
    max_eval = min(len(input_ids), 65536)

    print(f"\n{'='*70}\nEVALUATION\n{'='*70}")

    # Weight stats at end
    ws = student.weight_stats()
    print(f"  Final weight stats: {ws['confident_pct']:.1f}% confident, "
          f"{ws['uncertain_pct']:.1f}% uncertain", flush=True)

    for name, model, use_df in [('Teacher', teacher, False),
                                 ('Student+DF', student, True),
                                 ('Student only', student, False)]:
        total_loss = total_tok = 0
        for i in range(0, max_eval - 1, args.seq_len):
            chunk = input_ids[i:min(i + args.seq_len + 1, max_eval)].unsqueeze(0).to(device)
            if chunk.size(1) < 2: continue
            if use_df:
                logits = model(chunk[:, :-1], df_config=df_config).float()
            elif name == 'Teacher':
                logits = model(chunk[:, :-1]).float()
            else:
                logits = model(chunk[:, :-1], df_config=None).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   chunk[:, 1:].reshape(-1), reduction='sum')
            total_loss += loss.item(); total_tok += chunk.size(1) - 1
        ppl = math.exp(total_loss / max(total_tok, 1))
        print(f"  {name:<20s} PPL = {ppl:.2f}", flush=True)

    # Generation test
    print(f"\n  Generation test (with DF):")
    ids = tokenizer('The capital of France is', return_tensors='pt').input_ids.to(device)
    for _ in range(60):
        logits = student(ids, df_config=df_config).float()[0, -1]
        for prev_id in ids[0].tolist():
            if logits[prev_id] > 0: logits[prev_id] /= 1.2
            else: logits[prev_id] *= 1.2
        probs = torch.softmax(logits / 0.7, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        mask = cum - sorted_p > 0.9
        sorted_p[mask] = 0
        sorted_p /= sorted_p.sum()
        next_id = sorted_i[torch.multinomial(sorted_p, 1)].unsqueeze(0)
        ids = torch.cat([ids, next_id], dim=1)
    print(f"  {tokenizer.decode(ids[0])}", flush=True)


# ============================================================
# CLI
# ============================================================

def parse_tokens(s):
    s = s.upper().strip()
    if s.endswith('B'): return int(float(s[:-1]) * 1e9)
    elif s.endswith('M'): return int(float(s[:-1]) * 1e6)
    elif s.endswith('K'): return int(float(s[:-1]) * 1e3)
    return int(s)

def main():
    parser = argparse.ArgumentParser(description='DF-SSM Distillation with Density Field Weights')
    parser.add_argument('--model', default='state-spaces/mamba2-1.3b')
    parser.add_argument('--tokens', type=str, default='100M')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--block-len', type=int, default=64)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-dfw-mult', type=float, default=1.0,
                        help='LR multiplier for DFW latent weights (1.0 with linear clamp, no saturation)')
    parser.add_argument('--Ks', type=int, default=23, help='State density field K (23 for d_state=128)')
    parser.add_argument('--Kw', type=int, default=4, help='Weight density field K (4=17 levels, 8=65 levels)')
    parser.add_argument('--anneal-steps', type=int, default=2000,
                        help='Steps to anneal from continuous to fully quantized weights')
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--optimizer', type=str, default='schedulefree',
                        choices=['adamw', 'schedulefree'])
    parser.add_argument('--use-8bit-adam', action='store_true', default=True,
                        help='Use 8-bit Adam (saves ~5GB GPU memory)')
    parser.add_argument('--no-8bit-adam', dest='use_8bit_adam', action='store_false')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Warmup steps (default: min(1000, total_steps/10))')
    parser.add_argument('--quick', action='store_true', help='10M tokens')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP (FP32 training, more stable)')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Load model weights but reset optimizer (fresh fine-tuning)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (default: dfw_train_K_timestamp.log)')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    args.tokens = parse_tokens('10M' if args.quick else args.tokens)
    train(args)

if __name__ == '__main__':
    main()
