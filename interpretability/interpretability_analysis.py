#!/usr/bin/env python3
"""
Interpretability analysis:
1. Weight level histograms per layer
2. State visualization during text processing

Usage:
    python interpretability_analysis.py --scaffold dfssm_dfw_step1501.pt \
                                         --lora dfw_lora_all_r16_final.pt
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys, json
sys.path.insert(0, '.')
from df_ssm_dfw_lora import *
from transformers import AutoTokenizer


def load_model(scaffold_path, lora_path, device='cuda'):
    teacher, cfg, vocab_size = load_teacher('state-spaces/mamba2-1.3b', device='cpu')
    del teacher
    student = DFWMamba2LM(vocab_size=vocab_size, Ks=23, Kw=4, **cfg).to(device)
    ckpt = torch.load(scaffold_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'], strict=False)
    freeze_scaffold_add_lora(student, lora_layers='all', lora_rank=16, lora_targets='both')
    student = student.to(device)
    lora_ckpt = torch.load(lora_path, map_location=device, weights_only=False)
    for name, param in student.named_parameters():
        if name in lora_ckpt['lora_state']:
            param.data = lora_ckpt['lora_state'][name].to(device)
    student.eval()
    return student, cfg, vocab_size


# =====================================================================
# ANALYSIS 1: Weight level histograms per layer
# =====================================================================

def analyze_weight_levels(student):
    """Analyze distribution of 17 quantization levels across all layers."""
    print("=" * 70)
    print("WEIGHT LEVEL ANALYSIS")
    print("=" * 70)
    
    # 17 levels: 0/16, 1/16, ..., 16/16 → mapped to weights via (level*2-1)*scale
    # In frozen weights, the levels are: round(target * 16) / 16
    # where target = (clamp(latent, -1, 1) + 1) / 2
    # So quantized target ∈ {0/16, 1/16, ..., 16/16}
    # And weight = (quantized_target * 2 - 1) * scale
    # Normalized weight (before scale) ∈ {-1, -14/16, ..., 0, ..., 14/16, 1}
    
    all_histograms = {}
    layer_stats = []
    
    for name, m in student.named_modules():
        if isinstance(m, FrozenDFWWithLoRA):
            w = m.frozen_weight.data.cpu()
            scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
            w_norm = w / scale  # normalize to [-1, 1] range
            
            # Quantize to find which level each weight is at
            # w_norm should be at one of 17 levels: -1, -7/8, ..., 0, ..., 7/8, 1
            # Map to level index: level = round((w_norm + 1) / 2 * 16)
            levels = ((w_norm + 1) / 2 * 16).round().clamp(0, 16).long()
            
            # Histogram
            hist = torch.zeros(17, dtype=torch.long)
            for lv in range(17):
                hist[lv] = (levels == lv).sum().item()
            
            # Stats
            total = levels.numel()
            hist_pct = hist.float() / total * 100
            
            # Sparsity metrics
            at_zero = hist[8].item() / total * 100  # level 8 = 0.0
            at_extremes = (hist[0].item() + hist[16].item()) / total * 100  # ±1
            entropy = -(hist_pct[hist_pct > 0] / 100 * torch.log2(hist_pct[hist_pct > 0] / 100)).sum().item()
            
            layer_idx = name.split('.')[1] if 'layers.' in name else name
            proj_type = 'in_proj' if 'in_proj' in name else 'out_proj'
            key = f"L{layer_idx}_{proj_type}"
            
            all_histograms[key] = hist_pct.tolist()
            layer_stats.append({
                'layer': key,
                'at_zero': at_zero,
                'at_extremes': at_extremes,
                'entropy': entropy,
                'mean_level': levels.float().mean().item(),
                'std_level': levels.float().std().item(),
            })
            
    # Print summary
    print(f"\n{'Layer':<20} {'Zero%':>7} {'±1%':>7} {'Entropy':>8} {'Mean':>6} {'Std':>6}")
    print("-" * 60)
    for s in layer_stats:
        print(f"{s['layer']:<20} {s['at_zero']:>6.1f}% {s['at_extremes']:>6.1f}% "
              f"{s['entropy']:>8.2f} {s['mean_level']:>6.2f} {s['std_level']:>6.2f}")
    
    # Aggregate by layer position
    print("\n\nTRENDS BY LAYER DEPTH:")
    print("-" * 60)
    
    in_proj_stats = [s for s in layer_stats if 'in_proj' in s['layer']]
    out_proj_stats = [s for s in layer_stats if 'out_proj' in s['layer']]
    
    for group_name, group in [('in_proj', in_proj_stats), ('out_proj', out_proj_stats)]:
        n = len(group)
        early = group[:n//4]
        mid = group[n//4:3*n//4]
        late = group[3*n//4:]
        
        for region_name, region in [('Early (0-11)', early), ('Mid (12-35)', mid), ('Late (36-47)', late)]:
            avg_zero = np.mean([s['at_zero'] for s in region])
            avg_ext = np.mean([s['at_extremes'] for s in region])
            avg_ent = np.mean([s['entropy'] for s in region])
            avg_std = np.mean([s['std_level'] for s in region])
            print(f"  {group_name} {region_name:<15} "
                  f"zero={avg_zero:.1f}% ±1={avg_ext:.1f}% "
                  f"entropy={avg_ent:.2f} std={avg_std:.2f}")
    
    return all_histograms, layer_stats


# =====================================================================
# ANALYSIS 2: State visualization during text processing
# =====================================================================

def visualize_state(student, tokenizer, device='cuda'):
    """Capture and visualize SSM hidden state evolution during text processing."""
    print("\n\n" + "=" * 70)
    print("STATE VISUALIZATION")
    print("=" * 70)
    
    sentences = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "Once upon a time, in a land far away, there lived a brave knight.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    df = DensityFieldConfig(K=23, use_sigma_delta=True, sd_accumulator_bits=8, block_len=64)
    
    all_states = {}
    
    for sent in sentences:
        ids = tokenizer(sent, return_tensors='pt').input_ids.to(device)
        B, L = ids.shape
        
        # Pad to block_len
        block_len = 64
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            ids_padded = F.pad(ids, (pad_len, 0), value=tokenizer.eos_token_id or 0)
        else:
            ids_padded = ids
        
        # Hook into SSM layers to capture state
        captured_states = {}
        hooks = []
        
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                # Capture the intermediate after SSM scan
                # We'll capture the output of each mixer (which includes SSM state effects)
                captured_states[layer_name] = output.detach().cpu()
            return hook_fn
        
        # Attach hooks to a few layers (first, middle, last)
        target_layers = [0, 11, 23, 35, 47]
        for li in target_layers:
            layer = student.layers[li]
            h = layer.mixer.register_forward_hook(make_hook(f"layer_{li}"))
            hooks.append(h)
        
        with torch.no_grad():
            _ = student(ids_padded, df_config=df)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Analyze captured states
        tokens = tokenizer.convert_ids_to_tokens(ids[0])
        print(f"\nSentence: \"{sent}\"")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        state_summary = {}
        for layer_name, state in captured_states.items():
            # state is (B, L_padded, d_inner)
            if pad_len > 0:
                state = state[:, pad_len:, :]
            
            # Per-token statistics
            norms = state[0].norm(dim=-1)  # (L,)
            
            # Entropy of activation distribution (how spread out values are)
            state_abs = state[0].abs()
            state_hist = torch.histc(state_abs.flatten(), bins=50, min=0, max=state_abs.max())
            state_hist = state_hist / state_hist.sum()
            entropy = -(state_hist[state_hist > 0] * torch.log2(state_hist[state_hist > 0])).sum().item()
            
            # Sparsity (fraction near zero)
            threshold = state_abs.max() * 0.01
            sparsity = (state_abs < threshold).float().mean().item()
            
            # Token-to-token similarity (cosine of adjacent token representations)
            if state.shape[1] > 1:
                cos_sims = F.cosine_similarity(state[0, :-1], state[0, 1:], dim=-1)
                avg_cos = cos_sims.mean().item()
            else:
                avg_cos = 0.0
            
            state_summary[layer_name] = {
                'mean_norm': norms.mean().item(),
                'std_norm': norms.std().item(),
                'entropy': entropy,
                'sparsity': sparsity,
                'avg_cos_sim': avg_cos,
                'norm_per_token': norms.tolist()[:10],
            }
            
            print(f"  {layer_name}: norm={norms.mean():.2f}±{norms.std():.2f} "
                  f"entropy={entropy:.2f} sparsity={sparsity:.1%} cos_sim={avg_cos:.3f}")
        
        all_states[sent] = state_summary
    
    # Cross-sentence comparison
    print("\n\nCROSS-SENTENCE SIMILARITY (last layer, cosine):")
    print("-" * 60)
    
    # Re-run to get final layer output for each sentence
    sentence_vecs = {}
    for sent in sentences:
        ids = tokenizer(sent, return_tensors='pt').input_ids.to(device)
        B, L = ids.shape
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            ids_padded = F.pad(ids, (pad_len, 0), value=tokenizer.eos_token_id or 0)
        else:
            ids_padded = ids
        
        with torch.no_grad():
            output = student(ids_padded, df_config=df)
        
        # Use last token representation
        last_hidden = output[0, -1, :].cpu()
        sentence_vecs[sent] = last_hidden
    
    # Pairwise cosine similarity
    sents_short = [s[:40] + "..." if len(s) > 40 else s for s in sentences]
    print(f"\n{'':>45}", end="")
    for i in range(len(sentences)):
        print(f" {i:>5}", end="")
    print()
    
    for i, (s1, v1) in enumerate(sentence_vecs.items()):
        print(f"  {i}: {sents_short[i]:<42}", end="")
        for j, (s2, v2) in enumerate(sentence_vecs.items()):
            cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            print(f" {cos:>5.2f}", end="")
        print()
    
    # State pattern analysis across layers
    print("\n\nSTATE EVOLUTION ACROSS LAYERS:")
    print("-" * 60)
    print("How activation norms grow/shrink through the network:")
    
    for sent in sentences[:3]:
        summary = all_states[sent]
        norms = [summary[f"layer_{l}"]['mean_norm'] for l in target_layers]
        sparses = [summary[f"layer_{l}"]['sparsity'] for l in target_layers]
        print(f"\n  \"{sent[:50]}...\"")
        print(f"  Layer:    {' '.join(f'{l:>8}' for l in target_layers)}")
        print(f"  Norm:     {' '.join(f'{n:>8.2f}' for n in norms)}")
        print(f"  Sparse:   {' '.join(f'{s:>7.1%}' for s in sparses)}")
    
    return all_states


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scaffold', type=str, default='dfssm_dfw_step1501.pt')
    p.add_argument('--lora', type=str, default='dfw_lora_all_r16_final.pt')
    p.add_argument('--device', type=str, default='cuda')
    a = p.parse_args()
    
    student, cfg, vocab_size = load_model(a.scaffold, a.lora, a.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    
    # Analysis 1: Weight levels
    histograms, stats = analyze_weight_levels(student)
    
    # Analysis 2: State visualization
    states = visualize_state(student, tok, a.device)
    
    print("\n\nDone. Results above.")
