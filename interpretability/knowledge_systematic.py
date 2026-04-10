#!/usr/bin/env python3
"""
Systematic knowledge localization + concept codebook extraction.

Phase A: Run interventions on ALL pairs (not just France/Germany)
Phase B: Extract the "knowledge dimensions" at the flip layer
Phase C: Build a concept codebook — nearest-neighbor fact retrieval

Usage:
    python knowledge_systematic.py --scaffold dfssm_dfw_step1501.pt \
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


def forward_to_layer(model, input_ids, target_layer):
    """Run forward up to target_layer, return hidden state."""
    block_len = 64
    L = input_ids.shape[1]
    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        input_ids = F.pad(input_ids, (pad_len, 0), value=0)
    
    x = model.embedding(input_ids)
    for l in range(target_layer):
        x = x + model.layers[l].mixer(model.layers[l].norm(x))
    return x


def forward_from_layer(model, x, start_layer):
    """Continue forward from start_layer, return logits."""
    for l in range(start_layer, len(model.layers)):
        x = x + model.layers[l].mixer(model.layers[l].norm(x))
    x = model.norm_f(x)
    logits = x[:, -1, :] @ model.embedding.weight.T
    return logits


def get_hidden_at_all_layers(model, input_ids):
    """Get hidden state at the last token position for every layer."""
    block_len = 64
    L = input_ids.shape[1]
    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        input_ids = F.pad(input_ids, (pad_len, 0), value=0)
    
    states = []
    x = model.embedding(input_ids)
    states.append(x[:, -1, :].detach().cpu())
    
    for layer in model.layers:
        x = x + layer.mixer(layer.norm(x))
        states.append(x[:, -1, :].detach().cpu())
    
    return states  # len = n_layers + 1


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scaffold', default='dfssm_dfw_step1501.pt')
    p.add_argument('--lora', default='dfw_lora_all_r16_final.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    model, vocab_size = load_model(args.scaffold, args.lora, args.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    
    # ================================================================
    # FACT DATABASE
    # ================================================================
    
    # Same-template facts (capital cities)
    capitals = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of China is", "Beijing"),
        ("The capital of Brazil is", "Brasilia"),
        ("The capital of Australia is", "Canberra"),
        ("The capital of Canada is", "Ottawa"),
        ("The capital of Russia is", "Moscow"),
    ]
    
    # Different-template facts (for cross-domain comparison)
    other_facts = [
        ("The largest planet in the solar system is", "Jupiter"),
        ("The chemical symbol for water is", "H"),
        ("The speed of light in vacuum is approximately", "299"),
        ("The author of Romeo and Juliet is", "William"),
    ]
    
    # Get token IDs for expected answers
    for prompt, answer in capitals + other_facts:
        ans_ids = tok.encode(" " + answer, add_special_tokens=False)
        print(f"  \"{prompt}\" → {answer} (token {ans_ids[0]})")
    
    # ================================================================
    # PHASE A: Systematic intervention across all capital pairs
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE A: Intervention sweep — all capital pairs")
    print("=" * 70)
    
    # First, find the flip zone for each pair
    swap_layers = list(range(0, 48, 4))
    
    # Cache forward-to-layer results
    print("\n  Caching hidden states for all prompts...")
    cached_states = {}
    for prompt, answer in capitals:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        with torch.no_grad():
            states = get_hidden_at_all_layers(model, ids)
        cached_states[prompt] = states
        
        # Also cache forward-to-layer for intervention
        cached_fwd = {}
        for sl in swap_layers:
            with torch.no_grad():
                x = forward_to_layer(model, ids, sl)
            cached_fwd[sl] = x.detach()
        cached_states[prompt + "_fwd"] = cached_fwd
    
    # Test all pairs
    n_caps = len(capitals)
    flip_results = np.zeros((n_caps, n_caps, len(swap_layers)))
    
    print("\n  Running interventions...")
    for i, (prompt_a, ans_a) in enumerate(capitals):
        ans_a_id = tok.encode(" " + ans_a, add_special_tokens=False)[0]
        ids_a = tok(prompt_a, return_tensors='pt').input_ids.to(args.device)
        
        for j, (prompt_b, ans_b) in enumerate(capitals):
            if i == j:
                continue
            ans_b_id = tok.encode(" " + ans_b, add_special_tokens=False)[0]
            ids_b = tok(prompt_b, return_tensors='pt').input_ids.to(args.device)
            
            for si, sl in enumerate(swap_layers):
                with torch.no_grad():
                    # Get B's state at swap layer
                    x_b = forward_to_layer(model, ids_b, sl)
                    
                    # Get A's state at swap layer
                    x_a = forward_to_layer(model, ids_a, sl)
                    
                    # Swap last token
                    x_swapped = x_a.clone()
                    x_swapped[:, -1, :] = x_b[:, -1, :]
                    
                    # Continue forward
                    logits = forward_from_layer(model, x_swapped, sl)
                    probs = F.softmax(logits.float(), dim=-1)
                    
                    p_a = probs[0, ans_a_id].item()
                    p_b = probs[0, ans_b_id].item()
                    
                    # Store: positive = B dominates (flipped), negative = A dominates
                    flip_results[i, j, si] = p_b - p_a
    
    # Print flip zone summary
    print(f"\n  {'Pair':<30} {'Flip layer':>12} {'Max P(swap)':>12}")
    print(f"  {'-'*56}")
    
    flip_layers = []
    for i, (pa, aa) in enumerate(capitals):
        for j, (pb, ab) in enumerate(capitals):
            if i >= j:
                continue
            # Find first layer where B dominates when injected into A
            flip_layer = None
            for si, sl in enumerate(swap_layers):
                if flip_results[i, j, si] > 0:
                    flip_layer = sl
                    break
            
            max_flip = flip_results[i, j, :].max()
            label = f"{aa} → {ab}"
            fl_str = f"L{flip_layer}" if flip_layer is not None else "never"
            print(f"  {label:<30} {fl_str:>12} {max_flip:>12.4f}")
            if flip_layer is not None:
                flip_layers.append(flip_layer)
    
    if flip_layers:
        print(f"\n  Flip zone: L{min(flip_layers)}-L{max(flip_layers)} "
              f"(mean L{np.mean(flip_layers):.0f}, std {np.std(flip_layers):.1f})")
    
    # ================================================================
    # PHASE B: Knowledge dimensions at the flip layer
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PHASE B: Knowledge dimensions at the flip layer")
    print("=" * 70)
    
    # Use the mean flip layer
    knowledge_layer = int(np.mean(flip_layers)) if flip_layers else 30
    print(f"\n  Using layer L{knowledge_layer}")
    
    # Collect hidden states at the knowledge layer for all capitals
    knowledge_vectors = []
    knowledge_labels = []
    for prompt, answer in capitals:
        states = cached_states[prompt]
        # states[0] = after embedding, states[k+1] = after layer k
        vec = states[knowledge_layer + 1][0]  # +1 because states[0] is embedding
        knowledge_vectors.append(vec)
        knowledge_labels.append(answer)
    
    # Also collect for other-domain facts
    for prompt, answer in other_facts:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        with torch.no_grad():
            states = get_hidden_at_all_layers(model, ids)
        vec = states[knowledge_layer + 1][0]
        knowledge_vectors.append(vec)
        knowledge_labels.append(answer)
    
    K = torch.stack(knowledge_vectors)  # (N, d_model)
    
    # PCA to find the knowledge dimensions
    K_centered = K - K.mean(dim=0)
    U, S, V = torch.svd(K_centered.float())
    
    print(f"\n  PCA of hidden states at L{knowledge_layer}:")
    print(f"  {'Component':<12} {'Variance%':>10} {'Cumulative%':>12}")
    print(f"  {'-'*36}")
    total_var = (S ** 2).sum()
    cumulative = 0
    for c in range(min(20, len(S))):
        var_pct = (S[c] ** 2 / total_var * 100).item()
        cumulative += var_pct
        print(f"  PC{c:<9} {var_pct:>9.2f}% {cumulative:>11.2f}%")
        if cumulative > 95:
            print(f"  ... (95% variance captured in {c+1} components)")
            break
    
    # Project onto top 2 PCs for visualization
    projections = (K_centered.float() @ V[:, :2]).numpy()
    
    print(f"\n  2D projections (PC1, PC2):")
    print(f"  {'Concept':<15} {'PC1':>8} {'PC2':>8}")
    print(f"  {'-'*33}")
    for i, label in enumerate(knowledge_labels):
        print(f"  {label:<15} {projections[i, 0]:>8.2f} {projections[i, 1]:>8.2f}")
    
    # ================================================================
    # PHASE C: Concept codebook — nearest-neighbor retrieval
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PHASE C: Concept codebook — fact retrieval via state similarity")
    print("=" * 70)
    
    # Pairwise cosine similarity at the knowledge layer
    K_norm = F.normalize(K.float(), dim=-1)
    sim_matrix = (K_norm @ K_norm.T).numpy()
    
    print(f"\n  Cosine similarity at L{knowledge_layer}:")
    header = f"  {'':>15}" + "".join(f"{l:>8}" for l in knowledge_labels)
    print(header)
    for i, label in enumerate(knowledge_labels):
        row = f"  {label:>15}"
        for j in range(len(knowledge_labels)):
            val = sim_matrix[i, j]
            row += f" {val:>7.3f}"
        print(row)
    
    # Nearest-neighbor accuracy (within capitals)
    print(f"\n  Nearest-neighbor structure (capitals only):")
    n_caps = len(capitals)
    for i in range(n_caps):
        sims = [(sim_matrix[i, j], knowledge_labels[j]) for j in range(n_caps) if i != j]
        sims.sort(reverse=True)
        top3 = sims[:3]
        print(f"  {knowledge_labels[i]:>12} → nearest: {', '.join(f'{l}({s:.3f})' for s, l in top3)}")
    
    # ================================================================
    # PHASE D: Dimension reduction — how many dims carry the facts?
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PHASE D: How many dimensions encode factual knowledge?")
    print("=" * 70)
    
    # For each pair of capitals, find which dimensions differ most
    diff_dims_per_pair = []
    for i in range(n_caps):
        for j in range(i+1, n_caps):
            diff = (K[i] - K[j]).abs()
            # Top dimensions by absolute difference
            top_dims = diff.topk(50).indices.tolist()
            diff_dims_per_pair.append(set(top_dims))
    
    # How much overlap in the "important dimensions" across pairs?
    all_dims = set()
    for d in diff_dims_per_pair:
        all_dims.update(d)
    
    # Intersection: dimensions that appear in ALL pairs
    if diff_dims_per_pair:
        common_dims = diff_dims_per_pair[0]
        for d in diff_dims_per_pair[1:]:
            common_dims = common_dims.intersection(d)
    
        print(f"\n  Top-50 differing dimensions per pair:")
        print(f"    Total unique dimensions used: {len(all_dims)} / 2048")
        print(f"    Dimensions in ALL pairs:      {len(common_dims)} / 2048")
    
    # Test: can we distinguish capitals using only the common dimensions?
    if len(common_dims) > 0:
        common_list = sorted(common_dims)
        K_reduced = K[:n_caps, common_list]
        K_reduced_norm = F.normalize(K_reduced.float(), dim=-1)
        sim_reduced = (K_reduced_norm @ K_reduced_norm.T).numpy()
        
        print(f"\n  Similarity using ONLY {len(common_dims)} common dimensions:")
        for i in range(n_caps):
            sims = [(sim_reduced[i, j], knowledge_labels[j]) for j in range(n_caps) if i != j]
            sims.sort(reverse=True)
            print(f"  {knowledge_labels[i]:>12} → nearest: {sims[0][1]}({sims[0][0]:.3f})")
    
    # Reconstruction test: project onto top-K PCs, measure how much 
    # pairwise distance is preserved
    print(f"\n  Factual discriminability vs number of PCs:")
    print(f"  {'PCs':>6} {'Avg pairwise dist':>18} {'% of full':>10}")
    print(f"  {'-'*38}")
    
    full_dists = torch.cdist(K[:n_caps].float().unsqueeze(0), 
                              K[:n_caps].float().unsqueeze(0))[0]
    avg_full = full_dists[full_dists > 0].mean().item()
    
    for n_pcs in [2, 4, 8, 16, 32, 64, 128, 256]:
        K_proj = K_centered[:n_caps].float() @ V[:, :n_pcs]
        proj_dists = torch.cdist(K_proj.unsqueeze(0), K_proj.unsqueeze(0))[0]
        avg_proj = proj_dists[proj_dists > 0].mean().item()
        pct = avg_proj / avg_full * 100
        print(f"  {n_pcs:>6} {avg_proj:>18.2f} {pct:>9.1f}%")
    
    print(f"\n  Full dimensionality (2048): {avg_full:.2f}")
    
    # ================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if flip_layers:
        print(f"  Knowledge localization: L{min(flip_layers)}-L{max(flip_layers)}")
        print(f"  Tested: {len(flip_layers)} pairs, all flipped within this window")
        print(f"  Common knowledge dimensions: {len(common_dims) if diff_dims_per_pair else 'N/A'}")
        var_at_16 = (S[:16]**2).sum() / total_var * 100
        print(f"  Variance in 16 PCs: {var_at_16.item():.1f}%")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
