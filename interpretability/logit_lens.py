#!/usr/bin/env python3
"""
Logit lens — project hidden states through the LM head at every layer.
Shows what the model is "thinking about" at each processing stage.

Usage:
    python logit_lens.py --scaffold dfssm_dfw_step1501.pt \
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


def get_all_layer_states(model, input_ids):
    """Forward pass, return hidden state at last token for every layer."""
    block_len = 64
    L = input_ids.shape[1]
    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        input_ids = F.pad(input_ids, (pad_len, 0), value=0)
    
    states = []
    with torch.no_grad():
        x = model.embedding(input_ids)
        states.append(x[:, -1, :].clone())
        
        for layer in model.layers:
            x = x + layer.mixer(layer.norm(x))
            states.append(x[:, -1, :].clone())
        
        x_final = model.norm_f(x)
        states.append(x_final[:, -1, :].clone())
    
    return states  # len = n_layers + 2 (emb, L0..L47, norm_f)


def decode_top_k(hidden, emb_weight, tokenizer, k=10):
    """Project hidden state through LM head, return top-k tokens with probs."""
    logits = hidden.float() @ emb_weight.float().T
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_ids = probs.topk(k)
    tokens = [tokenizer.decode([t]).strip() for t in topk_ids.tolist()]
    probs_list = topk_probs.tolist()
    return list(zip(tokens, probs_list))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scaffold', default='dfssm_dfw_step1501.pt')
    p.add_argument('--lora', default='dfw_lora_all_r16_final.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    model, vocab_size = load_model(args.scaffold, args.lora, args.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    emb_weight = model.embedding.weight.data
    n_layers = len(model.layers)
    
    # Prompts to analyze
    prompts = [
        # Capitals
        "The capital of France is",
        "The capital of Germany is",
        # Language
        "The official language of France is",
        # Science
        "The chemical symbol for gold is",
        "The chemical symbol for water is",
        # People
        "The author of Romeo and Juliet is",
        "The theory of relativity was proposed by",
        # Common knowledge
        "The color of grass is",
        "The largest animal on Earth is the",
        # Different structure
        "Two plus two equals",
        "The company that created Windows is",
    ]
    
    # Layers to sample (every 4, plus key transitions)
    sample_layers = sorted(set([0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 
                                 30, 32, 34, 36, 40, 44, 47, n_layers]))
    sample_layers = [l for l in sample_layers if l <= n_layers]
    
    # ================================================================
    # PHASE 1: Full logit lens for each prompt
    # ================================================================
    print("=" * 90)
    print("LOGIT LENS — What does the model think at each layer?")
    print("=" * 90)
    
    for prompt in prompts:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        states = get_all_layer_states(model, ids)
        
        print(f"\n{'─' * 90}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"{'─' * 90}")
        print(f"  {'Layer':<8} {'Top-1':<15} {'P':>6} {'Top-2':<15} {'P':>6} "
              f"{'Top-3':<15} {'P':>6} {'Top-4':<12} {'Top-5':<12}")
        
        for li in sample_layers:
            top_k = decode_top_k(states[li][0], emb_weight, tok, k=5)
            
            layer_name = "emb" if li == 0 else (
                "norm_f" if li == n_layers else f"L{li-1}")
            
            row = f"  {layer_name:<8}"
            for token, prob in top_k:
                # Clean up token display
                token_clean = token[:12].replace('\n', '\\n')
                row += f" {token_clean:<14} {prob:>5.3f}"
            print(row)
    
    # ================================================================
    # PHASE 2: Track specific answer tokens through layers
    # ================================================================
    print(f"\n\n{'=' * 90}")
    print("ANSWER TRACKING — Rank and probability of correct answer at each layer")
    print(f"{'=' * 90}")
    
    tracked = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The official language of France is", "French"),
        ("The chemical symbol for water is", "H"),
        ("The author of Romeo and Juliet is", "William"),
        ("The color of grass is", "green"),
        ("Two plus two equals", "four"),
        ("The company that created Windows is", "Microsoft"),
    ]
    
    for prompt, answer in tracked:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        states = get_all_layer_states(model, ids)
        
        # Get answer token ID
        ans_ids = tok.encode(" " + answer, add_special_tokens=False)
        ans_id = ans_ids[0]
        
        print(f"\n  \"{prompt}\" → {answer} (token {ans_id})")
        print(f"  {'Layer':<8} {'Rank':>6} {'P(ans)':>8} {'Top-1':>15} {'P(top1)':>8}")
        
        prev_rank = None
        for li in sample_layers:
            logits = states[li][0].float() @ emb_weight.float().T
            probs = F.softmax(logits, dim=-1)
            
            ans_prob = probs[ans_id].item()
            rank = (probs > ans_prob).sum().item() + 1
            
            top1_id = probs.argmax().item()
            top1_tok = tok.decode([top1_id]).strip()[:12]
            top1_prob = probs[top1_id].item()
            
            layer_name = "emb" if li == 0 else (
                "norm_f" if li == n_layers else f"L{li-1}")
            
            # Mark significant rank changes
            marker = ""
            if prev_rank is not None:
                if rank < prev_rank * 0.5:
                    marker = " ↑↑"
                elif rank > prev_rank * 2:
                    marker = " ↓↓"
            prev_rank = rank
            
            print(f"  {layer_name:<8} {rank:>6} {ans_prob:>8.4f} {top1_tok:>15} {top1_prob:>8.3f}{marker}")
    
    # ================================================================
    # PHASE 3: Semantic trajectory — how does the "meaning" evolve?
    # ================================================================
    print(f"\n\n{'=' * 90}")
    print("SEMANTIC TRAJECTORY — Top predictions grouped by layer phase")
    print(f"{'=' * 90}")
    
    phases = [
        ("Embedding", [0]),
        ("Early (L0-L3)", [1, 2, 3, 4]),
        ("Build (L4-L15)", [5, 8, 12, 16]),
        ("Mid (L16-L27)", [17, 20, 24, 28]),
        ("Recall (L28-L35)", [29, 30, 32, 34, 36]),
        ("Format (L36-L47)", [37, 40, 44, 48]),
        ("Final norm", [n_layers]),
    ]
    
    for prompt in ["The capital of France is", 
                   "The chemical symbol for water is",
                   "The author of Romeo and Juliet is"]:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        states = get_all_layer_states(model, ids)
        
        print(f"\n  \"{prompt}\"")
        
        for phase_name, layer_indices in phases:
            # Collect unique top-5 tokens across layers in this phase
            all_tokens = {}
            for li in layer_indices:
                if li > n_layers:
                    continue
                top_k = decode_top_k(states[li][0], emb_weight, tok, k=5)
                for token, prob in top_k:
                    token_clean = token.replace('\n', '\\n').strip()
                    if token_clean and len(token_clean) > 1:
                        if token_clean not in all_tokens or prob > all_tokens[token_clean]:
                            all_tokens[token_clean] = prob
            
            # Sort by probability, show top unique tokens
            sorted_tokens = sorted(all_tokens.items(), key=lambda x: -x[1])[:8]
            token_str = ", ".join(f"{t}({p:.2f})" for t, p in sorted_tokens)
            print(f"    {phase_name:<20} {token_str}")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
