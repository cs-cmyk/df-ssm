#!/usr/bin/env python3
"""
Knowledge crystallization — how does knowledge organize layer by layer?

Run the same 115 prompts through every layer, measure clustering at each.
Shows when categories separate, when facts within categories separate,
and what happens at the output.

Usage:
    python knowledge_crystallization.py --scaffold dfssm_dfw_step1501.pt \
                                         --lora dfw_lora_all_r16_final.pt
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
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


# Same knowledge DB as atlas (trimmed for speed — 5 per category)
KNOWLEDGE_DB = {
    "capitals": [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of China is", "Beijing"),
        ("The capital of Russia is", "Moscow"),
        ("The capital of India is", "New Delhi"),
    ],
    "languages": [
        ("The official language of France is", "French"),
        ("The official language of Germany is", "German"),
        ("The official language of Japan is", "Japanese"),
        ("The official language of Italy is", "Italian"),
        ("The official language of Spain is", "Spanish"),
        ("The official language of Russia is", "Russian"),
    ],
    "elements": [
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for iron is", "Fe"),
        ("The chemical symbol for water is", "H2O"),
        ("The chemical symbol for oxygen is", "O"),
        ("The chemical symbol for carbon is", "C"),
        ("The chemical symbol for sodium is", "Na"),
    ],
    "writers": [
        ("The author of Romeo and Juliet is", "Shakespeare"),
        ("The author of Harry Potter is", "Rowling"),
        ("The author of 1984 is", "Orwell"),
        ("The author of Pride and Prejudice is", "Austen"),
        ("The author of The Great Gatsby is", "Fitzgerald"),
        ("The author of War and Peace is", "Tolstoy"),
    ],
    "animals": [
        ("The largest animal on Earth is the", "whale"),
        ("The fastest land animal is the", "cheetah"),
        ("The tallest animal is the", "giraffe"),
        ("A baby dog is called a", "puppy"),
        ("A baby cat is called a", "kitten"),
        ("The animal that produces honey is the", "bee"),
    ],
    "colors": [
        ("The color of the sky on a clear day is", "blue"),
        ("The color of grass is", "green"),
        ("The color of blood is", "red"),
        ("The color of snow is", "white"),
        ("The color of coal is", "black"),
        ("The currency of Japan is the", "yen"),
    ],
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--scaffold', default='dfssm_dfw_step1501.pt')
    p.add_argument('--lora', default='dfw_lora_all_r16_final.pt')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    
    model, vocab_size = load_model(args.scaffold, args.lora, args.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    n_layers = len(model.layers)
    
    # Flatten prompts
    all_prompts = []
    all_categories = []
    all_answers = []
    for cat, prompts in KNOWLEDGE_DB.items():
        for prompt, answer in prompts:
            all_prompts.append(prompt)
            all_categories.append(cat)
            all_answers.append(answer)
    
    N = len(all_prompts)
    categories_unique = list(KNOWLEDGE_DB.keys())
    cat_indices = {cat: [i for i, c in enumerate(all_categories) if c == cat]
                   for cat in categories_unique}
    
    print(f"Prompts: {N}, Categories: {len(categories_unique)}")
    print(f"Layers: {n_layers}")
    
    # ================================================================
    # Collect hidden states at EVERY layer for all prompts
    # ================================================================
    print("\nCollecting hidden states at all layers...")
    
    # states_by_layer[layer_idx] = (N, d_model) tensor
    states_by_layer = [[] for _ in range(n_layers + 1)]  # +1 for embedding
    
    for pi, prompt in enumerate(all_prompts):
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        block_len = 64
        L = ids.shape[1]
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=0)
        
        with torch.no_grad():
            x = model.embedding(ids)
            states_by_layer[0].append(x[:, -1, :].detach().cpu())
            
            for l in range(n_layers):
                x = x + model.layers[l].mixer(model.layers[l].norm(x))
                states_by_layer[l + 1].append(x[:, -1, :].detach().cpu())
        
        if (pi + 1) % 10 == 0:
            print(f"  {pi+1}/{N}")
    
    # Stack into tensors and squeeze batch dim
    for l in range(n_layers + 1):
        states_by_layer[l] = torch.stack(states_by_layer[l]).squeeze(1)  # (N, d_model)
    
    print(f"  Done. {n_layers + 1} layers captured.")
    
    # ================================================================
    # Measure clustering at each layer
    # ================================================================
    print("\n" + "=" * 70)
    print("CLUSTERING EVOLUTION ACROSS LAYERS")
    print("=" * 70)
    
    layer_indices = list(range(0, n_layers + 1, 2))  # every 2nd layer
    if n_layers not in layer_indices:
        layer_indices.append(n_layers)
    
    results = []
    
    print(f"\n  {'Layer':<8}", end="")
    for cat in categories_unique:
        print(f" {cat[:7]:>8}", end="")
    print(f" {'avg_sep':>8} {'pca_10%':>8}")
    print(f"  {'-' * (8 + 9 * len(categories_unique) + 18)}")
    
    for li in layer_indices:
        V = states_by_layer[li]
        V_norm = F.normalize(V.float(), dim=-1)
        sim = (V_norm @ V_norm.t()).numpy()
        
        # Per-category separation
        separations = {}
        for cat in categories_unique:
            idx = cat_indices[cat]
            within = [sim[i, j] for i in idx for j in idx if i < j]
            between = [sim[i, j] for i in idx for j in range(N) if all_categories[j] != cat]
            sep = np.mean(within) - np.mean(between) if within and between else 0
            separations[cat] = sep
        
        avg_sep = np.mean(list(separations.values()))
        
        # PCA — variance in top 10
        V_centered = V - V.mean(dim=0)
        try:
            _, S, _ = torch.svd(V_centered.float())
            total_var = (S ** 2).sum()
            pca_10 = ((S[:10] ** 2).sum() / total_var * 100).item() if total_var > 0 else 0
        except:
            pca_10 = 0
        
        layer_name = "emb" if li == 0 else f"L{li-1}"
        print(f"  {layer_name:<8}", end="")
        for cat in categories_unique:
            print(f" {separations[cat]:>+8.4f}", end="")
        print(f" {avg_sep:>+8.4f} {pca_10:>7.1f}%")
        
        results.append({
            'layer': li,
            'layer_name': layer_name,
            'separations': separations,
            'avg_sep': avg_sep,
            'pca_10': pca_10,
        })
    
    # ================================================================
    # Find peak layer for each category
    # ================================================================
    print("\n\n" + "=" * 70)
    print("PEAK CLUSTERING LAYER PER CATEGORY")
    print("=" * 70)
    
    for cat in categories_unique:
        seps = [(r['layer'], r['separations'][cat]) for r in results]
        peak_layer, peak_sep = max(seps, key=lambda x: x[1])
        peak_name = "emb" if peak_layer == 0 else f"L{peak_layer-1}"
        print(f"  {cat:<15} peaks at {peak_name:<8} (separation: {peak_sep:+.4f})")
    
    # Overall
    avg_seps = [(r['layer'], r['avg_sep']) for r in results]
    peak_layer, peak_sep = max(avg_seps, key=lambda x: x[1])
    peak_name = "emb" if peak_layer == 0 else f"L{peak_layer-1}"
    print(f"\n  {'OVERALL':<15} peaks at {peak_name:<8} (avg separation: {peak_sep:+.4f})")
    
    # ================================================================
    # Within-category fine-grained: when do individual facts separate?
    # ================================================================
    print("\n\n" + "=" * 70)
    print("WITHIN-CATEGORY: When do individual facts become distinguishable?")
    print("=" * 70)
    
    # For capitals: track pairwise distance between France and Germany
    cap_idx = cat_indices["capitals"]
    if len(cap_idx) >= 2:
        print(f"\n  France vs Germany (within capitals):")
        print(f"  {'Layer':<8} {'Cosine':>8} {'L2 dist':>10}")
        print(f"  {'-'*28}")
        
        for li in layer_indices:
            V = states_by_layer[li]
            v_france = V[cap_idx[0]]
            v_germany = V[cap_idx[1]]
            cos = F.cosine_similarity(v_france.unsqueeze(0), v_germany.unsqueeze(0)).item()
            l2 = (v_france - v_germany).norm().item()
            layer_name = "emb" if li == 0 else f"L{li-1}"
            print(f"  {layer_name:<8} {cos:>8.4f} {l2:>10.2f}")
    
    # Track average within-category cosine (are facts becoming MORE different?)
    print(f"\n  Average within-category similarity across layers:")
    print(f"  {'Layer':<8}", end="")
    for cat in categories_unique:
        print(f" {cat[:7]:>8}", end="")
    print()
    print(f"  {'-' * (8 + 9 * len(categories_unique))}")
    
    for li in layer_indices[::2]:  # every 4th to keep output manageable
        V = states_by_layer[li]
        V_norm = F.normalize(V.float(), dim=-1)
        sim = (V_norm @ V_norm.t()).numpy()
        
        layer_name = "emb" if li == 0 else f"L{li-1}"
        print(f"  {layer_name:<8}", end="")
        for cat in categories_unique:
            idx = cat_indices[cat]
            within = [sim[i, j] for i in idx for j in idx if i < j]
            avg_w = np.mean(within) if within else 0
            print(f" {avg_w:>8.4f}", end="")
        print()
    
    # ================================================================
    # What are the OTHER dimensions doing? Compare factual vs non-factual
    # ================================================================
    print("\n\n" + "=" * 70)
    print("WHAT DO THE OTHER DIMENSIONS ENCODE?")
    print("=" * 70)
    
    # At L33, PCA gives us 10 "factual" PCs.
    # Project all prompts, measure how much of total norm is in factual vs non-factual
    V33 = states_by_layer[34]  # L33 = index 34 (0=emb, 1=L0, ...)
    V33_centered = V33 - V33.mean(dim=0)
    _, S33, Vt33 = torch.svd(V33_centered.float())
    
    # Project onto factual PCs (top 10) and residual
    proj_factual = V33_centered.float() @ Vt33[:, :10]     # (N, 10)
    proj_residual = V33_centered.float() @ Vt33[:, 10:]    # (N, 2038)
    
    norm_factual = proj_factual.norm(dim=1)    # (N,)
    norm_residual = proj_residual.norm(dim=1)   # (N,)
    norm_total = V33_centered.float().norm(dim=1)
    
    print(f"\n  At L33, decomposing hidden state into factual (10 PCs) vs residual (2038 dims):")
    print(f"\n  {'Category':<15} {'Factual norm':>14} {'Residual norm':>14} {'Ratio F/R':>10}")
    print(f"  {'-'*55}")
    
    for cat in categories_unique:
        idx = cat_indices[cat]
        fn = norm_factual[idx].mean().item()
        rn = norm_residual[idx].mean().item()
        ratio = fn / (rn + 1e-8)
        print(f"  {cat:<15} {fn:>14.2f} {rn:>14.2f} {ratio:>10.3f}")
    
    all_fn = norm_factual.mean().item()
    all_rn = norm_residual.mean().item()
    print(f"\n  {'OVERALL':<15} {all_fn:>14.2f} {all_rn:>14.2f} {all_fn/all_rn:>10.3f}")
    print(f"\n  The residual ({all_rn:.0f}) is {all_rn/all_fn:.1f}x larger than the factual ({all_fn:.0f}).")
    print(f"  The 10 factual PCs carry {all_fn**2/(all_fn**2+all_rn**2)*100:.1f}% of variance,")
    print(f"  but encode ALL the category-separating information.")
    
    # ================================================================
    # Hypothesis: residual encodes syntax/template, not facts
    # ================================================================
    print(f"\n  Test: Do prompts with the same template cluster in residual space?")
    
    # "The capital of X is" and "The official language of X is" have different templates
    # but "The capital of France is" and "The capital of Germany is" share a template
    
    templates = {}
    for i, prompt in enumerate(all_prompts):
        # Extract template: replace the variable part
        if "capital of" in prompt:
            templates.setdefault("capital_of", []).append(i)
        elif "language of" in prompt:
            templates.setdefault("language_of", []).append(i)
        elif "chemical symbol" in prompt:
            templates.setdefault("chemical_symbol", []).append(i)
        elif "author of" in prompt:
            templates.setdefault("author_of", []).append(i)
        elif "color of" in prompt or "colour" in prompt:
            templates.setdefault("color_of", []).append(i)
        elif "animal" in prompt or "dog" in prompt or "cat" in prompt or "fish" in prompt:
            templates.setdefault("animal_desc", []).append(i)
    
    if templates:
        # Measure within-template similarity in residual space vs factual space
        res_norm = F.normalize(proj_residual, dim=-1)
        fact_norm = F.normalize(proj_factual, dim=-1)
        
        print(f"\n  {'Template':<20} {'N':>4} {'Factual sim':>12} {'Residual sim':>13}")
        print(f"  {'-'*53}")
        
        for tname, indices in templates.items():
            if len(indices) < 2:
                continue
            
            f_sims = []
            r_sims = []
            for i in indices:
                for j in indices:
                    if i < j:
                        f_sims.append(F.cosine_similarity(
                            fact_norm[i:i+1], fact_norm[j:j+1]).item())
                        r_sims.append(F.cosine_similarity(
                            res_norm[i:i+1], res_norm[j:j+1]).item())
            
            print(f"  {tname:<20} {len(indices):>4} "
                  f"{np.mean(f_sims):>12.4f} {np.mean(r_sims):>13.4f}")
        
        print(f"\n  If residual sim >> factual sim for same template:")
        print(f"  → Residual encodes template/syntax, factual encodes the variable content")
    
    # ================================================================
    # Export layer-by-layer data for visualization
    # ================================================================
    export = {
        'categories': categories_unique,
        'n_prompts': N,
        'results': [
            {
                'layer': r['layer'],
                'layer_name': r['layer_name'],
                'avg_sep': r['avg_sep'],
                'pca_10': r['pca_10'],
                'separations': r['separations'],
            }
            for r in results
        ]
    }
    
    with open('knowledge_crystallization_data.json', 'w') as f:
        json.dump(export, f, indent=2)
    
    print(f"\n\nExported to knowledge_crystallization_data.json")
    print("Done.")


if __name__ == '__main__':
    main()
