#!/usr/bin/env python3
"""
Knowledge localization in DF-SSM.

Three-phase experiment:
  1. LOCATE:     Where do "France" and "Germany" paths diverge?
  2. PROBE:      At which layer does "Paris"/"Berlin" appear as prediction?
  3. INTERVENE:  Swap hidden states at divergence layer — does prediction flip?

Usage:
    python knowledge_localization.py --scaffold dfssm_dfw_step1501.pt \
                                      --lora dfw_lora_all_r16_final.pt
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
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
    return student, vocab_size


def get_per_layer_hidden_states(model, input_ids, device='cuda'):
    """Run forward pass, capturing hidden state after each layer's residual add."""
    B, L = input_ids.shape
    block_len = 64
    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        input_ids = F.pad(input_ids, (pad_len, 0), value=0)
    
    x = model.embedding(input_ids)
    
    hidden_states = [x[:, -1, :].detach().cpu()]  # after embedding, last token
    
    for layer in model.layers:
        x = x + layer.mixer(layer.norm(x))
        hidden_states.append(x[:, -1, :].detach().cpu())  # last token after this layer
    
    x = model.norm_f(x)
    hidden_states.append(x[:, -1, :].detach().cpu())  # after final norm
    
    return hidden_states  # list of (B, d_model) tensors, length = n_layers + 2


def probe_at_layer(hidden_state, embedding_weight):
    """Project hidden state through LM head, return top-k predictions."""
    # hidden_state: (d_model,)
    # embedding_weight: (vocab_size, d_model) — tied weights
    logits = hidden_state.float() @ embedding_weight.float().T
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = probs.topk(10)
    return topk_ids.tolist(), topk_probs.tolist()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scaffold', default='dfssm_dfw_step1501.pt')
    p.add_argument('--lora', default='dfw_lora_all_r16_final.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    model, vocab_size = load_model(args.scaffold, args.lora, args.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    emb_weight = model.embedding.weight.data.cpu()
    
    # ================================================================
    # EXPERIMENT SET 1: Capital cities
    # ================================================================
    pairs = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
    ]
    
    # Also test control pairs (same structure, different domain)
    controls = [
        ("The largest planet in the solar system is", "Jupiter"),
        ("The chemical symbol for water is", "H"),
        ("The programming language created by Guido van Rossum is", "Python"),
    ]
    
    all_prompts = pairs + controls
    
    print("=" * 70)
    print("PHASE 1: LOCATE — Where do parallel prompts diverge?")
    print("=" * 70)
    
    # Get hidden states for all prompts
    all_hidden = {}
    for prompt, expected in all_prompts:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        hidden = get_per_layer_hidden_states(model, ids, args.device)
        all_hidden[prompt] = hidden
        
        # Check final prediction
        top_ids, top_probs = probe_at_layer(hidden[-1][0], emb_weight)
        top_tokens = [tok.decode([t]).strip() for t in top_ids]
        print(f"\n  \"{prompt}\"")
        print(f"    Expected: {expected}")
        print(f"    Top-5: {', '.join(f'{t}({p:.3f})' for t, p in zip(top_tokens[:5], top_probs[:5]))}")
    
    # Pairwise cosine similarity between France and Germany at each layer
    print("\n\n" + "=" * 70)
    print("PHASE 1b: Layer-by-layer divergence")
    print("=" * 70)
    
    prompt_pairs = [
        ("The capital of France is", "The capital of Germany is", "France vs Germany"),
        ("The capital of France is", "The capital of Japan is", "France vs Japan"),
        ("The capital of France is", "The largest planet in the solar system is", "France vs Jupiter (control)"),
    ]
    
    for prompt_a, prompt_b, label in prompt_pairs:
        h_a = all_hidden[prompt_a]
        h_b = all_hidden[prompt_b]
        
        print(f"\n  {label}:")
        print(f"    {'Layer':<8} {'Cosine':>8} {'L2 dist':>10} {'Divergence':>12}")
        print(f"    {'-'*42}")
        
        prev_cos = 1.0
        for layer_idx in range(0, len(h_a), max(1, len(h_a)//12)):
            cos = F.cosine_similarity(h_a[layer_idx][0:1], h_b[layer_idx][0:1]).item()
            l2 = (h_a[layer_idx][0] - h_b[layer_idx][0]).norm().item()
            delta = prev_cos - cos
            marker = " <<<" if delta > 0.05 else ""
            
            layer_name = f"emb" if layer_idx == 0 else (
                f"norm_f" if layer_idx == len(h_a)-1 else f"L{layer_idx-1}")
            print(f"    {layer_name:<8} {cos:>8.4f} {l2:>10.2f} {delta:>+11.4f}{marker}")
            prev_cos = cos
    
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PHASE 2: PROBE — When does the answer crystallize?")
    print("=" * 70)
    
    target_prompts = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
    ]
    
    for prompt, expected in target_prompts:
        hidden = all_hidden[prompt]
        expected_id = tok.encode(" " + expected, add_special_tokens=False)[0]
        
        print(f"\n  \"{prompt}\" → expecting \"{expected}\" (token {expected_id})")
        print(f"    {'Layer':<8} {'Top-1':>12} {'P(top1)':>8} {'P(expected)':>12} {'Rank':>6}")
        print(f"    {'-'*52}")
        
        for layer_idx in range(0, len(hidden), max(1, len(hidden)//12)):
            top_ids, top_probs = probe_at_layer(hidden[layer_idx][0], emb_weight)
            top1_token = tok.decode([top_ids[0]]).strip()
            
            # Find rank of expected token
            logits = hidden[layer_idx][0].float() @ emb_weight.float().T
            probs = F.softmax(logits, dim=-1)
            expected_prob = probs[expected_id].item()
            rank = (probs > expected_prob).sum().item() + 1
            
            layer_name = f"emb" if layer_idx == 0 else (
                f"norm_f" if layer_idx == len(hidden)-1 else f"L{layer_idx-1}")
            
            marker = " <<<" if top_ids[0] == expected_id else ""
            print(f"    {layer_name:<8} {top1_token:>12} {top_probs[0]:>8.3f} {expected_prob:>12.4f} {rank:>6}{marker}")
    
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PHASE 3: INTERVENE — Swap hidden states, does prediction flip?")
    print("=" * 70)
    
    prompt_a = "The capital of France is"
    prompt_b = "The capital of Germany is"
    ids_a = tok(prompt_a, return_tensors='pt').input_ids.to(args.device)
    ids_b = tok(prompt_b, return_tensors='pt').input_ids.to(args.device)
    
    # Find the key divergence layer from Phase 1
    h_a_all = all_hidden[prompt_a]
    h_b_all = all_hidden[prompt_b]
    
    # Compute cosine at each layer to find divergence point
    cos_per_layer = []
    for i in range(len(h_a_all)):
        cos = F.cosine_similarity(h_a_all[i][0:1], h_b_all[i][0:1]).item()
        cos_per_layer.append(cos)
    
    # Find layer with biggest drop
    max_drop = 0
    divergence_layer = 0
    for i in range(1, len(cos_per_layer)):
        drop = cos_per_layer[i-1] - cos_per_layer[i]
        if drop > max_drop:
            max_drop = drop
            divergence_layer = i - 1  # layer index (0-based)
    
    print(f"\n  Divergence layer: L{divergence_layer} (cosine drop: {max_drop:.4f})")
    
    # Run intervention at multiple layers
    test_layers = sorted(set([0, divergence_layer, divergence_layer+1, 
                               len(model.layers)//2, len(model.layers)-1]))
    
    for swap_layer in test_layers:
        if swap_layer >= len(model.layers):
            continue
            
        # Run France prompt but swap hidden state at swap_layer with Germany's
        block_len = 64
        
        # Forward pass A (France) with intervention
        ids_a_padded = F.pad(ids_a, (block_len - ids_a.shape[1] % block_len, 0), value=0) \
            if ids_a.shape[1] % block_len != 0 else ids_a
        ids_b_padded = F.pad(ids_b, (block_len - ids_b.shape[1] % block_len, 0), value=0) \
            if ids_b.shape[1] % block_len != 0 else ids_b
        
        with torch.no_grad():
            # Get Germany's hidden state at swap_layer
            x_b = model.embedding(ids_b_padded)
            for l in range(swap_layer):
                x_b = x_b + model.layers[l].mixer(model.layers[l].norm(x_b))
            germany_state = x_b.clone()
            
            # Run France forward, inject Germany's state at swap_layer
            x_a = model.embedding(ids_a_padded)
            for l in range(swap_layer):
                x_a = x_a + model.layers[l].mixer(model.layers[l].norm(x_a))
            
            # SWAP: replace France's state with Germany's at last token position
            x_swapped = x_a.clone()
            x_swapped[:, -1, :] = germany_state[:, -1, :]
            
            # Continue forward from swap point
            for l in range(swap_layer, len(model.layers)):
                x_swapped = x_swapped + model.layers[l].mixer(model.layers[l].norm(x_swapped))
            x_swapped = model.norm_f(x_swapped)
            
            logits = x_swapped[:, -1, :] @ model.embedding.weight.T
            probs = F.softmax(logits.float(), dim=-1)
            top5_ids = probs[0].topk(5).indices.tolist()
            top5_probs = probs[0].topk(5).values.tolist()
            top5_tokens = [tok.decode([t]).strip() for t in top5_ids]
        
        paris_id = tok.encode(" Paris", add_special_tokens=False)[0]
        berlin_id = tok.encode(" Berlin", add_special_tokens=False)[0]
        p_paris = probs[0, paris_id].item()
        p_berlin = probs[0, berlin_id].item()
        
        print(f"\n  Swap at L{swap_layer} (France prompt, Germany state injected):")
        print(f"    Top-5: {', '.join(f'{t}({p:.3f})' for t, p in zip(top5_tokens, top5_probs))}")
        print(f"    P(Paris)={p_paris:.4f}  P(Berlin)={p_berlin:.4f}  "
              f"{'FLIPPED!' if p_berlin > p_paris else 'not flipped'}")
    
    # ================================================================
    # BONUS: Full swap matrix — try swapping at every 4th layer
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PHASE 3b: Swap at every 4th layer — P(Paris) vs P(Berlin)")
    print("=" * 70)
    print(f"\n  {'Layer':<8} {'P(Paris)':>10} {'P(Berlin)':>10} {'Prediction':>12} {'Flipped?':>10}")
    print(f"  {'-'*54}")
    
    # Baseline (no swap)
    with torch.no_grad():
        x_base = model.embedding(ids_a_padded)
        for l in range(len(model.layers)):
            x_base = x_base + model.layers[l].mixer(model.layers[l].norm(x_base))
        x_base = model.norm_f(x_base)
        logits_base = x_base[:, -1, :] @ model.embedding.weight.T
        probs_base = F.softmax(logits_base.float(), dim=-1)
        p_paris_base = probs_base[0, paris_id].item()
        p_berlin_base = probs_base[0, berlin_id].item()
        top1_base = tok.decode([probs_base[0].argmax().item()]).strip()
    
    print(f"  {'none':<8} {p_paris_base:>10.4f} {p_berlin_base:>10.4f} {top1_base:>12} {'(baseline)':>10}")
    
    for swap_layer in range(0, len(model.layers), 4):
        with torch.no_grad():
            # Get Germany state at swap_layer
            x_b = model.embedding(ids_b_padded)
            for l in range(swap_layer):
                x_b = x_b + model.layers[l].mixer(model.layers[l].norm(x_b))
            
            # Run France, swap at swap_layer
            x_a = model.embedding(ids_a_padded)
            for l in range(swap_layer):
                x_a = x_a + model.layers[l].mixer(model.layers[l].norm(x_a))
            
            x_a[:, -1, :] = x_b[:, -1, :]
            
            for l in range(swap_layer, len(model.layers)):
                x_a = x_a + model.layers[l].mixer(model.layers[l].norm(x_a))
            x_a = model.norm_f(x_a)
            
            logits = x_a[:, -1, :] @ model.embedding.weight.T
            probs = F.softmax(logits.float(), dim=-1)
            p_p = probs[0, paris_id].item()
            p_b = probs[0, berlin_id].item()
            top1 = tok.decode([probs[0].argmax().item()]).strip()
            flipped = "YES" if p_b > p_p else "no"
        
        print(f"  L{swap_layer:<6} {p_p:>10.4f} {p_b:>10.4f} {top1:>12} {flipped:>10}")
    
    print("\n\nDone.")


if __name__ == '__main__':
    main()
