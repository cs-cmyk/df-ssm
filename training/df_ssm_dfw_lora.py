#!/usr/bin/env python3
"""
DF-SSM DFW + LoRA: Freeze quantized scaffold, train low-rank correction.

Usage:
  # Last layer out_proj only (130K params, fastest test)
  python df_ssm_dfw_lora.py --resume dfssm_dfw_step1501.pt --lora-layers last1

  # Last 4 layers (1M params)
  python df_ssm_dfw_lora.py --resume dfssm_dfw_step1501.pt --lora-layers last4

  # Last 8 layers (2M params)
  python df_ssm_dfw_lora.py --resume dfssm_dfw_step1501.pt --lora-layers last8

  # All layers
  python df_ssm_dfw_lora.py --resume dfssm_dfw_step1501.pt --lora-layers all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys
import argparse
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

# Import model components from DFW script
from df_ssm_mamba2_distill_dfw import (
    DFWMamba2LM, DensityFieldLinear, DensityFieldConfig,
    SprayBank, STEClamp, DFWQuantize, RMSNorm,
    DFWMamba2Block, DFWMamba2ResidualBlock,
    TeacherMamba2Block, TeacherMamba2ResidualBlock, TeacherMamba2LM,
    load_teacher, get_data_iterator, df_ssd_scan, ssd_scan_no_df,
)


# ============================================================
# Frozen DFW Linear with LoRA
# ============================================================

class FrozenDFWWithLoRA(nn.Module):
    """DensityFieldLinear frozen at deterministic quantized levels + trainable LoRA."""
    
    def __init__(self, dfw_linear: DensityFieldLinear, rank=16, lora_alpha=1.0):
        super().__init__()
        out_features = dfw_linear.out_features
        in_features = dfw_linear.in_features
        Kw = dfw_linear.Kw
        KK = Kw * Kw
        
        # Freeze: deterministic quantization to nearest level (no spray rotation)
        with torch.no_grad():
            target_density = (dfw_linear.latent_weight.data.clamp(-1, 1) + 1) / 2
            # Round to nearest of K²+1 levels
            quantized = (target_density * KK).round() / KK
            effective_w = quantized * 2.0 - 1.0
            frozen_weight = effective_w * dfw_linear.scale.data.unsqueeze(1)
        
        self.register_buffer('frozen_weight', frozen_weight)
        
        # Bias (frozen)
        if dfw_linear.bias is not None:
            self.register_buffer('frozen_bias', dfw_linear.bias.data.clone())
        else:
            self.frozen_bias = None
        
        # LoRA: low-rank correction
        # Initialize B to zero so LoRA starts with zero contribution
        self.lora_A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_alpha = lora_alpha
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # Store metadata
        self.out_features = out_features
        self.in_features = in_features
        self.Kw = Kw
    
    def forward(self, x):
        # Frozen scaffold (no gradients)
        main_out = F.linear(x, self.frozen_weight, self.frozen_bias)
        
        # LoRA correction (trainable)
        lora_out = F.linear(F.linear(x, self.lora_B), self.lora_A) * self.scaling
        
        return main_out + lora_out
    
    @torch.no_grad()
    def lora_stats(self):
        """Report LoRA contribution magnitude."""
        lora_weight = (self.lora_A @ self.lora_B) * self.scaling
        scaffold_norm = self.frozen_weight.norm().item()
        lora_norm = lora_weight.norm().item()
        ratio = lora_norm / (scaffold_norm + 1e-8)
        return {
            'scaffold_norm': scaffold_norm,
            'lora_norm': lora_norm,
            'lora_ratio': ratio * 100,  # percentage
        }


def freeze_scaffold_add_lora(student, lora_layers='last1', lora_rank=16,
                               lora_targets='both', lora_alpha=1.0):
    """
    Replace DensityFieldLinear layers with FrozenDFWWithLoRA.
    
    lora_layers: 'last1', 'last4', 'last8', 'last16', 'all'
    lora_targets: 'out_proj', 'in_proj', 'both'
    """
    n_layers = len(student.layers)
    
    # Determine which layers get LoRA
    if lora_layers == 'all':
        lora_layer_indices = set(range(n_layers))
    elif lora_layers.startswith('last'):
        n = int(lora_layers[4:])
        lora_layer_indices = set(range(n_layers - n, n_layers))
    else:
        raise ValueError(f"Unknown lora_layers: {lora_layers}")
    
    total_lora_params = 0
    total_frozen_params = 0
    layers_modified = 0
    
    for i, layer in enumerate(student.layers):
        mixer = layer.mixer
        
        if i in lora_layer_indices:
            # Add LoRA to this layer
            if lora_targets in ('out_proj', 'both'):
                old = mixer.out_proj
                new = FrozenDFWWithLoRA(old, rank=lora_rank, lora_alpha=lora_alpha)
                mixer.out_proj = new
                total_lora_params += lora_rank * (old.out_features + old.in_features)
                total_frozen_params += old.out_features * old.in_features
                
            if lora_targets in ('in_proj', 'both'):
                old = mixer.in_proj
                new = FrozenDFWWithLoRA(old, rank=lora_rank, lora_alpha=lora_alpha)
                mixer.in_proj = new
                total_lora_params += lora_rank * (old.out_features + old.in_features)
                total_frozen_params += old.out_features * old.in_features
            
            # Freeze non-LoRA DFW layers in this block
            for name, mod in mixer.named_modules():
                if isinstance(mod, DensityFieldLinear):
                    freeze_dfw_layer(mod)
            
            layers_modified += 1
        else:
            # Freeze ALL DFW layers in non-LoRA blocks
            for name, mod in mixer.named_modules():
                if isinstance(mod, DensityFieldLinear):
                    freeze_dfw_layer(mod)
    
    # Freeze all non-LoRA parameters
    for name, param in student.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
    
    return total_lora_params, total_frozen_params, layers_modified


def freeze_dfw_layer(dfw_linear):
    """Freeze a DensityFieldLinear by replacing forward with deterministic quantization."""
    Kw = dfw_linear.Kw
    KK = Kw * Kw
    
    with torch.no_grad():
        target_density = (dfw_linear.latent_weight.data.clamp(-1, 1) + 1) / 2
        quantized = (target_density * KK).round() / KK
        effective_w = quantized * 2.0 - 1.0
        frozen_weight = effective_w * dfw_linear.scale.data.unsqueeze(1)
    
    # Replace forward with simple frozen linear
    dfw_linear.register_buffer('_frozen_weight', frozen_weight)
    
    def frozen_forward(self, x):
        return F.linear(x, self._frozen_weight, self.bias)
    
    import types
    dfw_linear.forward = types.MethodType(frozen_forward, dfw_linear)
    dfw_linear.latent_weight.requires_grad = False
    dfw_linear.scale.requires_grad = False


# ============================================================
# Training
# ============================================================

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_file = args.log_file or f'dfw_lora_{args.lora_layers}_r{args.lora_rank}_{int(time.time())}.log'
    
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
    print("DF-SSM DFW + LoRA CORRECTION TRAINING")
    print("=" * 70)
    print(f"  DFW checkpoint:  {args.resume}")
    print(f"  LoRA layers:     {args.lora_layers}")
    print(f"  LoRA targets:    {args.lora_targets}")
    print(f"  LoRA rank:       {args.lora_rank}")
    print(f"  LoRA alpha:      {args.lora_alpha}")
    print(f"  Tokens:          {args.tokens:,}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Grad accum:      {args.grad_accum}")
    print(f"  LR:              {args.lr}")
    print(f"  Kw:              {args.Kw}")
    
    # Load teacher
    teacher, cfg, vocab_size = load_teacher(args.model, device='cpu')
    teacher = teacher.to(device).half()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Load student from DFW checkpoint
    print(f"\n  Loading DFW student from {args.resume}...")
    student = DFWMamba2LM(vocab_size=vocab_size, Ks=args.Ks, Kw=args.Kw, **cfg).to(device)
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    missing, unexpected = student.load_state_dict(ckpt['model_state_dict'], strict=False)
    spray_missing = [k for k in missing if 'spray_bank' in k]
    real_missing = [k for k in missing if 'spray_bank' not in k]
    if real_missing:
        print(f"  WARNING: missing keys: {real_missing[:5]}")
    if spray_missing:
        print(f"  SprayBank buffers skipped ({len(spray_missing)})")
    
    # Freeze scaffold, add LoRA
    print(f"\n  Freezing scaffold, adding LoRA...")
    lora_params, frozen_params, layers_modified = freeze_scaffold_add_lora(
        student, 
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_targets=args.lora_targets,
        lora_alpha=args.lora_alpha,
    )
    # Move new LoRA modules to device (they were created on CPU)
    student = student.to(device)
    
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    
    print(f"  Scaffold frozen:   {frozen_params:,} params (deterministic 17 levels)")
    print(f"  LoRA trainable:    {lora_params:,} params ({lora_params/1e6:.2f}M)")
    print(f"  LoRA memory:       {lora_params * 4 / 1e6:.1f} MB (FP32)")
    print(f"  Total trainable:   {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    print(f"  Layers with LoRA:  {layers_modified}")
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    
    # Density field config for state
    df_config = DensityFieldConfig(K=args.Ks, use_sigma_delta=True,
                                    sd_accumulator_bits=8, block_len=args.block_len)
    
    # Optimizer — only LoRA params
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    
    total_steps = args.tokens // (args.seq_len * args.batch_size * args.grad_accum)
    warmup_steps = min(100, total_steps // 10)
    
    def lr_schedule(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    data_iter = get_data_iterator(tokenizer, seq_len=args.seq_len,
                                    batch_size=args.batch_size, device=device)
    
    # CSV log
    csv_file = log_file.replace('.log', '.csv')
    csv_f = open(csv_file, 'w')
    csv_f.write('step,tokens_M,loss,lm_loss,kl_loss,ppl,lr,lora_ratio_pct\n')
    csv_f.flush()
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    print(f"\n{'='*70}\nTRAINING\n{'='*70}")
    print(f"  Total steps: {total_steps}, warmup: {warmup_steps}")
    
    # Quick pre-LoRA eval
    student.eval()
    pre_loss = 0
    pre_tok = 0
    eval_iter = get_data_iterator(tokenizer, seq_len=args.seq_len,
                                    batch_size=args.batch_size, device=device)
    with torch.no_grad():
        for i, (ids, tgt) in enumerate(eval_iter):
            if i >= 10: break
            logits = student(ids, df_config=df_config).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            pre_loss += loss.item() * ids.numel()
            pre_tok += ids.numel()
    pre_ppl = math.exp(pre_loss / pre_tok)
    print(f"  Pre-LoRA PPL:  {pre_ppl:.1f}")
    del eval_iter
    
    student.train()
    total_tokens = 0
    step = 0
    accum_loss = accum_lm = accum_kl = 0
    start_time = time.time()
    optimizer.zero_grad()
    
    # Interrupt handler
    interrupted = [False]
    import signal
    def handle_interrupt(signum, frame):
        if interrupted[0]: raise SystemExit(1)
        interrupted[0] = True
        print(f"\n  Ctrl+C caught. Saving...", flush=True)
    signal.signal(signal.SIGINT, handle_interrupt)
    
    use_amp = (device == 'cuda' and not args.no_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    amp_str = "AMP" if use_amp else "FP32"
    print(f"  Precision: {amp_str}")
    
    for batch_idx, (input_ids, targets) in enumerate(data_iter):
        if interrupted[0]:
            save_lora_checkpoint(student, optimizer, scheduler, step, total_tokens, args)
            csv_f.close()
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
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: NaN/Inf at batch {batch_idx}, skipping", flush=True)
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in student.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            
            if step % 5 == 0:
                avg_loss = accum_loss / (5 * args.grad_accum)
                avg_lm = accum_lm / (5 * args.grad_accum)
                avg_kl = accum_kl / (5 * args.grad_accum)
                elapsed = time.time() - start_time
                ppl = math.exp(min(avg_lm, 20))
                lr_now = scheduler.get_last_lr()[0]
                mem = f" | {torch.cuda.max_memory_allocated()/1e9:.1f}GB" if device == 'cuda' else ""
                print(f"  Step {step:>6d} | Loss {avg_loss:.4f} (LM {avg_lm:.4f} KL {avg_kl:.4f}) | "
                      f"PPL {ppl:.1f} | LR {lr_now:.2e} | {amp_str} | "
                      f"{total_tokens/elapsed:.0f} tok/s | {total_tokens/1e6:.1f}M{mem}", flush=True)
                
                # CSV
                lora_ratio = 0
                for m in student.modules():
                    if isinstance(m, FrozenDFWWithLoRA):
                        stats = m.lora_stats()
                        lora_ratio = stats['lora_ratio']
                        break
                csv_f.write(f'{step},{total_tokens/1e6:.2f},{avg_loss:.4f},{avg_lm:.4f},'
                            f'{avg_kl:.4f},{ppl:.1f},{lr_now:.6f},{lora_ratio:.4f}\n')
                csv_f.flush()
                
                accum_loss = accum_lm = accum_kl = 0
            
            # LoRA stats every 100 steps
            if step % 100 == 0:
                lora_norms = []
                for m in student.modules():
                    if isinstance(m, FrozenDFWWithLoRA):
                        lora_norms.append(m.lora_stats())
                if lora_norms:
                    avg_ratio = np.mean([s['lora_ratio'] for s in lora_norms])
                    max_ratio = max(s['lora_ratio'] for s in lora_norms)
                    print(f"    LoRA: avg={avg_ratio:.2f}% of scaffold, max={max_ratio:.2f}%",
                          flush=True)
            
            if step % 1000 == 0:
                save_lora_checkpoint(student, optimizer, scheduler, step, total_tokens, args)
        
        if total_tokens >= args.tokens: break
    
    elapsed = time.time() - start_time
    print(f"\n  Done: {total_tokens/1e6:.1f}M tokens, {step} steps, {elapsed/3600:.1f}h")
    csv_f.close()
    
    save_lora_checkpoint(student, optimizer, scheduler, step, total_tokens, args, final=True)
    
    # Final eval
    print(f"\n{'='*70}\nEVALUATION\n{'='*70}")
    student.eval()
    eval_ppl = evaluate_ppl(student, tokenizer, df_config, device, args)
    print(f"  Pre-LoRA PPL:   {pre_ppl:.1f}")
    print(f"  Post-LoRA PPL:  {eval_ppl:.1f}")
    print(f"  Improvement:    {pre_ppl:.1f} → {eval_ppl:.1f} ({(1-eval_ppl/pre_ppl)*100:.1f}%)")
    
    # LoRA stats
    print(f"\n  LoRA weight statistics:")
    for i, layer in enumerate(student.layers):
        mixer = layer.mixer
        for name in ['in_proj', 'out_proj']:
            mod = getattr(mixer, name)
            if isinstance(mod, FrozenDFWWithLoRA):
                stats = mod.lora_stats()
                print(f"    Layer {i} {name}: LoRA = {stats['lora_ratio']:.3f}% of scaffold")


def evaluate_ppl(student, tokenizer, df_config, device, args):
    """Evaluate PPL on WikiText-2."""
    try:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n'.join(t for t in [item['text'] for item in ds] if t.strip())
        ids = tokenizer(text, return_tensors='pt', truncation=False).input_ids[0]
    except:
        return float('inf')
    
    total_loss = total_tok = 0
    with torch.no_grad():
        for i in range(0, min(len(ids), 65536) - 1, args.seq_len):
            chunk = ids[i:i + args.seq_len + 1].unsqueeze(0).to(device)
            if chunk.size(1) < 2: continue
            logits = student(chunk[:, :-1], df_config=df_config).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    chunk[:, 1:].reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tok += chunk.size(1) - 1
    
    return math.exp(total_loss / total_tok)


def save_lora_checkpoint(student, optimizer, scheduler, step, total_tokens, args, final=False):
    """Save only LoRA weights (tiny)."""
    suffix = 'final' if final else f'step{step}'
    
    # Extract only LoRA state
    lora_state = {}
    for name, param in student.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data
    
    path = f'dfw_lora_{args.lora_layers}_r{args.lora_rank}_{suffix}.pt'
    torch.save({
        'lora_state': lora_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'step': step,
        'total_tokens': total_tokens,
        'args': vars(args),
        'lora_layers': args.lora_layers,
        'lora_rank': args.lora_rank,
    }, path)
    
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved: {path} ({size_mb:.1f} MB)", flush=True)


def parse_tokens(s):
    s = s.strip().upper()
    if s.endswith('M'): return int(float(s[:-1]) * 1e6)
    if s.endswith('B'): return int(float(s[:-1]) * 1e9)
    return int(s)


def main():
    parser = argparse.ArgumentParser(description='DFW + LoRA correction training')
    parser.add_argument('--resume', type=str, required=True, help='DFW checkpoint to load')
    parser.add_argument('--model', type=str, default='state-spaces/mamba2-1.3b')
    parser.add_argument('--tokens', type=str, default='20M')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--block-len', type=int, default=64)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3, help='LoRA learning rate (higher than base)')
    parser.add_argument('--Ks', type=int, default=23)
    parser.add_argument('--Kw', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--lora-layers', type=str, default='last1',
                        choices=['last1', 'last2', 'last4', 'last8', 'last16', 'all'])
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-targets', type=str, default='both',
                        choices=['out_proj', 'in_proj', 'both'])
    parser.add_argument('--lora-alpha', type=float, default=1.0)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--log-file', type=str, default=None)
    args = parser.parse_args()
    args.tokens = parse_tokens(args.tokens)
    train(args)


if __name__ == '__main__':
    main()
