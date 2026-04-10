#!/usr/bin/env python3
"""
Is early-layer space a "categorical intent" space?

Test: train a nearest-centroid classifier at each layer.
If L3 achieves >90% category accuracy but L3's logit lens gives garbage,
then L3 operates in a categorical space distinct from vocabulary space.

Also: what ARE the categories? Use contrastive projection to find
what vocabulary words best describe each category centroid.

Usage:
    python category_space.py --scaffold dfssm_dfw_step1501.pt \
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
        ("The capital of Egypt is", "Cairo"),
        ("The capital of Canada is", "Ottawa"),
    ],
    "languages": [
        ("The official language of France is", "French"),
        ("The official language of Germany is", "German"),
        ("The official language of Japan is", "Japanese"),
        ("The official language of Italy is", "Italian"),
        ("The official language of Spain is", "Spanish"),
        ("The official language of Russia is", "Russian"),
        ("The official language of China is", "Chinese"),
        ("The official language of Brazil is", "Portuguese"),
    ],
    "elements": [
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for iron is", "Fe"),
        ("The chemical symbol for water is", "H2O"),
        ("The chemical symbol for oxygen is", "O"),
        ("The chemical symbol for carbon is", "C"),
        ("The chemical symbol for sodium is", "Na"),
        ("The chemical symbol for copper is", "Cu"),
        ("The chemical symbol for helium is", "He"),
    ],
    "writers": [
        ("The author of Romeo and Juliet is", "Shakespeare"),
        ("The author of Harry Potter is", "Rowling"),
        ("The author of 1984 is", "Orwell"),
        ("The author of Pride and Prejudice is", "Austen"),
        ("The author of The Great Gatsby is", "Fitzgerald"),
        ("The author of War and Peace is", "Tolstoy"),
        ("The author of Don Quixote is", "Cervantes"),
        ("The author of Les Miserables is", "Hugo"),
    ],
    "animals": [
        ("The largest animal on Earth is the", "whale"),
        ("The fastest land animal is the", "cheetah"),
        ("The tallest animal is the", "giraffe"),
        ("The largest bird is the", "ostrich"),
        ("A baby dog is called a", "puppy"),
        ("A baby cat is called a", "kitten"),
        ("The animal that produces honey is the", "bee"),
        ("A group of lions is called a", "pride"),
    ],
    "colors": [
        ("The color of the sky on a clear day is", "blue"),
        ("The color of grass is", "green"),
        ("The color of blood is", "red"),
        ("The color of snow is", "white"),
        ("The color of coal is", "black"),
        ("The currency of Japan is the", "yen"),
        ("The currency of the United Kingdom is the", "pound"),
        ("The currency of the United States is the", "dollar"),
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
    emb_weight = model.embedding.weight.data.cpu()
    n_layers = len(model.layers)
    
    # Flatten
    all_prompts, all_cats, all_answers = [], [], []
    for cat, prompts in KNOWLEDGE_DB.items():
        for prompt, answer in prompts:
            all_prompts.append(prompt)
            all_cats.append(cat)
            all_answers.append(answer)
    
    N = len(all_prompts)
    cats_unique = list(KNOWLEDGE_DB.keys())
    cat_to_idx = {c: i for i, c in enumerate(cats_unique)}
    labels = np.array([cat_to_idx[c] for c in all_cats])
    
    # Collect hidden states at all layers
    print(f"Collecting {N} prompts × {n_layers+1} layers...")
    states = [[] for _ in range(n_layers + 1)]
    
    for prompt in all_prompts:
        ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
        block_len = 64
        L = ids.shape[1]
        pad_len = (block_len - L % block_len) % block_len
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=0)
        
        with torch.no_grad():
            x = model.embedding(ids)
            states[0].append(x[:, -1, :].squeeze(0).cpu())
            for l in range(n_layers):
                x = x + model.layers[l].mixer(model.layers[l].norm(x))
                states[l+1].append(x[:, -1, :].squeeze(0).cpu())
    
    for l in range(n_layers + 1):
        states[l] = torch.stack(states[l])
    
    # ================================================================
    # TEST 1: Nearest-centroid classification at each layer
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Category classification accuracy (leave-one-out)")
    print("=" * 70)
    
    print(f"\n  {'Layer':<8} {'Accuracy':>10} {'Logit lens':>12}  Notes")
    print(f"  {'-'*50}")
    
    for li in list(range(0, n_layers+1, 2)):
        V = states[li]
        V_norm = F.normalize(V.float(), dim=-1)
        
        # Leave-one-out nearest centroid
        correct = 0
        for i in range(N):
            # Centroid of each category excluding sample i
            best_cat = -1
            best_sim = -1
            for ci, cat in enumerate(cats_unique):
                members = [j for j in range(N) if all_cats[j] == cat and j != i]
                if not members:
                    continue
                centroid = V_norm[members].mean(dim=0)
                centroid = F.normalize(centroid.unsqueeze(0), dim=-1)
                sim = F.cosine_similarity(V_norm[i:i+1], centroid).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cat = ci
            if best_cat == labels[i]:
                correct += 1
        
        acc = correct / N * 100
        
        # Logit lens quality: does top-1 make sense?
        top1_tokens = set()
        for i in range(min(5, N)):
            logits = V[i].float() @ emb_weight.float().T
            top1_id = logits.argmax().item()
            top1_tokens.add(tok.decode([top1_id]).strip()[:10])
        
        lens_quality = ", ".join(list(top1_tokens)[:3])
        
        layer_name = "emb" if li == 0 else f"L{li-1}"
        notes = ""
        if acc > 90:
            notes = "← STRONG"
        elif acc > 70:
            notes = "← good"
        
        print(f"  {layer_name:<8} {acc:>9.1f}% {lens_quality:>12}  {notes}")
    
    # ================================================================
    # TEST 2: Contrastive projection — what DO the categories look like?
    # ================================================================
    print("\n\n" + "=" * 70)
    print("TEST 2: What vocabulary words describe each category centroid?")
    print("=" * 70)
    
    for target_layer in [0, 4, 16, 34, n_layers]:
        V = states[target_layer]
        layer_name = "emb" if target_layer == 0 else (
            f"L{target_layer-1}" if target_layer <= n_layers else "norm_f")
        
        print(f"\n  Layer {layer_name}:")
        
        # Compute category centroids
        centroids = {}
        for cat in cats_unique:
            idx = [i for i, c in enumerate(all_cats) if c == cat]
            centroids[cat] = V[idx].float().mean(dim=0)
        
        # Global centroid (mean of everything)
        global_centroid = V.float().mean(dim=0)
        
        for cat in cats_unique:
            # Contrastive: category centroid - global centroid
            diff = centroids[cat] - global_centroid
            
            # Project through LM head
            logits = diff @ emb_weight.float().T
            top_ids = logits.topk(10).indices.tolist()
            top_vals = logits.topk(10).values.tolist()
            top_tokens = [tok.decode([t]).strip() for t in top_ids]
            
            # Also bottom (anti-correlated words)
            bot_ids = logits.topk(5, largest=False).indices.tolist()
            bot_tokens = [tok.decode([t]).strip() for t in bot_ids]
            
            print(f"    {cat:<12} → {', '.join(t for t in top_tokens[:6] if len(t) > 1)}")
            print(f"    {'':12}   anti: {', '.join(t for t in bot_tokens[:4] if len(t) > 1)}")
    
    # ================================================================
    # TEST 3: Category-to-category transitions through layers
    # ================================================================
    print("\n\n" + "=" * 70)
    print("TEST 3: How does category representation evolve?")
    print("=" * 70)
    
    # For one prompt, show its similarity to each category centroid at each layer
    test_prompts = [
        ("The capital of France is", "capitals"),
        ("The chemical symbol for gold is", "elements"),
        ("The author of Romeo and Juliet is", "writers"),
    ]
    
    for prompt, true_cat in test_prompts:
        pi = all_prompts.index(prompt)
        
        print(f"\n  \"{prompt}\" (true: {true_cat})")
        print(f"  {'Layer':<8}", end="")
        for cat in cats_unique:
            print(f" {cat[:7]:>8}", end="")
        print(f" {'pred':>8}")
        print(f"  {'-'*(8 + 9*len(cats_unique) + 9)}")
        
        for li in [0, 2, 4, 8, 16, 24, 34, n_layers]:
            V = states[li]
            V_norm = F.normalize(V.float(), dim=-1)
            
            # Centroid for each category (excluding this sample)
            sims = {}
            for cat in cats_unique:
                idx = [j for j in range(N) if all_cats[j] == cat and j != pi]
                if not idx:
                    sims[cat] = 0
                    continue
                centroid = F.normalize(V_norm[idx].mean(dim=0, keepdim=True), dim=-1)
                sims[cat] = F.cosine_similarity(V_norm[pi:pi+1], centroid).item()
            
            pred = max(sims, key=sims.get)
            layer_name = "emb" if li == 0 else (
                f"L{li-1}" if li <= n_layers else "norm_f")
            
            print(f"  {layer_name:<8}", end="")
            for cat in cats_unique:
                marker = "*" if cat == pred else " "
                print(f" {sims[cat]:>7.3f}{marker}", end="")
            print(f" {pred:>8}")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
