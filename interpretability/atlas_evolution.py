#!/usr/bin/env python3
"""
Four atlases — same prompts, four representational spaces.
Shows how categories rearrange as they move through the network.

Usage:
    python atlas_evolution.py --scaffold dfssm_dfw_step1501.pt \
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


PROMPTS = {
    "capitals": [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The capital of China is", "Beijing"),
        ("The capital of Russia is", "Moscow"),
    ],
    "languages": [
        ("The official language of France is", "French"),
        ("The official language of Germany is", "German"),
        ("The official language of Japan is", "Japanese"),
        ("The official language of Spain is", "Spanish"),
        ("The official language of Russia is", "Russian"),
        ("The official language of China is", "Chinese"),
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
    ],
}

TARGET_LAYERS = [3, 15, 33, 47]  # intent, translation, knowledge, output


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
    
    # Flatten
    all_prompts, all_cats, all_labels = [], [], []
    for cat, prompts in PROMPTS.items():
        for prompt, label in prompts:
            all_prompts.append(prompt)
            all_cats.append(cat)
            all_labels.append(label)
    
    N = len(all_prompts)
    cats_unique = list(PROMPTS.keys())
    
    # Collect states at target layers
    print(f"Collecting {N} prompts at layers {TARGET_LAYERS}...")
    
    layer_states = {l: [] for l in TARGET_LAYERS}
    
    for prompt in all_prompts:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        block_len = 64
        L = ids.shape[1]
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=0)
        
        with torch.no_grad():
            x = model.embedding(ids)
            for l in range(n_layers):
                x = x + model.layers[l].mixer(model.layers[l].norm(x))
                if (l + 1) in TARGET_LAYERS:  # l+1 because layer 0 output = after first layer
                    layer_states[l + 1].append(x[:, -1, :].squeeze(0).cpu())
    
    for l in TARGET_LAYERS:
        layer_states[l] = torch.stack(layer_states[l])
    
    # For each target layer: PCA, clustering, export
    export = {}
    
    for tl in TARGET_LAYERS:
        V = layer_states[tl]
        V_centered = V - V.mean(dim=0)
        
        _, S, Vt = torch.svd(V_centered.float())
        proj = (V_centered.float() @ Vt[:, :2]).numpy()
        
        # Clustering metrics
        V_norm = F.normalize(V.float(), dim=-1)
        sim = (V_norm @ V_norm.t()).numpy()
        
        total_var = (S ** 2).sum()
        var_2pc = ((S[:2] ** 2).sum() / total_var * 100).item()
        
        layer_name = f"L{tl-1}" if tl <= n_layers else "norm_f"
        space_name = {3: "Intent", 15: "Translation", 33: "Knowledge", 47: "Output"}[tl]
        
        print(f"\n{'='*60}")
        print(f"LAYER {layer_name} — {space_name} space")
        print(f"{'='*60}")
        print(f"  2 PCs capture {var_2pc:.1f}% of variance")
        
        # Per-category stats
        separations = []
        for cat in cats_unique:
            idx = [i for i, c in enumerate(all_cats) if c == cat]
            within = [sim[i, j] for i in idx for j in idx if i < j]
            between = [sim[i, j] for i in idx for j in range(N) if all_cats[j] != cat]
            sep = np.mean(within) - np.mean(between)
            separations.append(sep)
            
            pts = proj[idx]
            cx, cy = pts.mean(axis=0)
            spread = np.sqrt(((pts - pts.mean(axis=0))**2).sum(axis=1).mean())
            
            print(f"  {cat:<12} centroid=({cx:>7.1f},{cy:>7.1f}) spread={spread:>5.1f} sep={sep:>+.3f}")
            print(f"    {'  '.join(f'{all_labels[i]}({proj[i,0]:.0f},{proj[i,1]:.0f})' for i in idx[:4])}")
        
        avg_sep = np.mean(separations)
        print(f"  Average separation: {avg_sep:+.3f}")
        
        # Export
        export[f"L{tl-1}"] = {
            'space': space_name,
            'var_2pc': var_2pc,
            'avg_sep': avg_sep,
            'points': [
                {'label': all_labels[i], 'category': all_cats[i],
                 'x': float(proj[i, 0]), 'y': float(proj[i, 1])}
                for i in range(N)
            ]
        }
    
    with open('atlas_evolution.json', 'w') as f:
        json.dump(export, f, indent=2)
    
    print(f"\n\nExported to atlas_evolution.json")
    print("Done.")


if __name__ == '__main__':
    main()
