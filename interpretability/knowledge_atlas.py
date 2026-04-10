#!/usr/bin/env python3
"""
Knowledge Atlas — mine the model's knowledge subspace.

Run ~200 factual prompts through to the knowledge layer (L33),
project onto the low-dimensional subspace, and map the layout.

Usage:
    python knowledge_atlas.py --scaffold dfssm_dfw_step1501.pt \
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


def get_state_at_layer(model, input_ids, target_layer, device='cuda'):
    """Forward to target_layer, return last-token hidden state."""
    block_len = 64
    L = input_ids.shape[1]
    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        input_ids = F.pad(input_ids, (pad_len, 0), value=0)
    
    with torch.no_grad():
        x = model.embedding(input_ids)
        for l in range(target_layer + 1):
            x = x + model.layers[l].mixer(model.layers[l].norm(x))
        return x[:, -1, :].detach().cpu()


# ================================================================
# FACTUAL PROMPT DATABASE
# ================================================================

KNOWLEDGE_DB = {
    # Geography — capitals
    "capitals": [
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
        ("The capital of India is", "New Delhi"),
        ("The capital of Egypt is", "Cairo"),
        ("The capital of Mexico is", "Mexico City"),
        ("The capital of South Korea is", "Seoul"),
        ("The capital of Turkey is", "Ankara"),
    ],
    
    # Geography — continents
    "continents": [
        ("France is located in", "Europe"),
        ("Japan is located in", "Asia"),
        ("Brazil is located in", "South America"),
        ("Australia is located in", "Oceania"),
        ("Egypt is located in", "Africa"),
        ("Canada is located in", "North America"),
        ("India is located in", "Asia"),
        ("Germany is located in", "Europe"),
        ("China is located in", "Asia"),
        ("Mexico is located in", "North America"),
    ],
    
    # Geography — languages
    "languages": [
        ("The official language of France is", "French"),
        ("The official language of Germany is", "German"),
        ("The official language of Japan is", "Japanese"),
        ("The official language of Italy is", "Italian"),
        ("The official language of Spain is", "Spanish"),
        ("The official language of China is", "Chinese"),
        ("The official language of Brazil is", "Portuguese"),
        ("The official language of Russia is", "Russian"),
        ("The official language of Turkey is", "Turkish"),
        ("The official language of South Korea is", "Korean"),
    ],
    
    # Science — elements
    "elements": [
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for silver is", "Ag"),
        ("The chemical symbol for iron is", "Fe"),
        ("The chemical symbol for water is", "H2O"),
        ("The chemical symbol for oxygen is", "O"),
        ("The chemical symbol for carbon is", "C"),
        ("The chemical symbol for sodium is", "Na"),
        ("The chemical symbol for helium is", "He"),
        ("The chemical symbol for nitrogen is", "N"),
        ("The chemical symbol for copper is", "Cu"),
    ],
    
    # Science — physics
    "physics": [
        ("The speed of light is approximately", "300000"),
        ("The boiling point of water is", "100"),
        ("The freezing point of water is", "0"),
        ("The acceleration due to gravity is approximately", "9.8"),
        ("The number of planets in the solar system is", "eight"),
        ("The largest planet in the solar system is", "Jupiter"),
        ("The smallest planet in the solar system is", "Mercury"),
        ("The closest star to Earth is", "the Sun"),
        ("The chemical formula for salt is", "NaCl"),
        ("The speed of sound in air is approximately", "343"),
    ],
    
    # People — famous scientists
    "scientists": [
        ("The theory of relativity was proposed by", "Albert Einstein"),
        ("The laws of motion were formulated by", "Isaac Newton"),
        ("The theory of evolution was proposed by", "Charles Darwin"),
        ("The discoverer of penicillin was", "Alexander Fleming"),
        ("The inventor of the telephone was", "Alexander Graham Bell"),
        ("The inventor of the light bulb was", "Thomas Edison"),
        ("The discoverer of radium was", "Marie Curie"),
        ("The father of computer science is", "Alan Turing"),
        ("The creator of the World Wide Web is", "Tim Berners"),
        ("The author of A Brief History of Time is", "Stephen Hawking"),
    ],
    
    # People — writers
    "writers": [
        ("The author of Romeo and Juliet is", "William Shakespeare"),
        ("The author of Harry Potter is", "J.K. Rowling"),
        ("The author of 1984 is", "George Orwell"),
        ("The author of Pride and Prejudice is", "Jane Austen"),
        ("The author of The Great Gatsby is", "F. Scott Fitzgerald"),
        ("The author of Don Quixote is", "Miguel de Cervantes"),
        ("The author of War and Peace is", "Leo Tolstoy"),
        ("The author of The Odyssey is", "Homer"),
        ("The author of Hamlet is", "William Shakespeare"),
        ("The author of Les Miserables is", "Victor Hugo"),
    ],
    
    # Technology — programming
    "programming": [
        ("The programming language created by Guido van Rossum is", "Python"),
        ("The programming language created by James Gosling is", "Java"),
        ("The programming language created by Bjarne Stroustrup is", "C++"),
        ("The programming language created by Dennis Ritchie is", "C"),
        ("The operating system created by Linus Torvalds is", "Linux"),
        ("The company that created Windows is", "Microsoft"),
        ("The company that created the iPhone is", "Apple"),
        ("The company that created Android is", "Google"),
        ("The company that created Facebook is", "Meta"),
        ("The founder of Amazon is", "Jeff Bezos"),
    ],
    
    # Math — numbers
    "numbers": [
        ("The square root of 144 is", "12"),
        ("The value of pi is approximately", "3.14"),
        ("Two plus two equals", "four"),
        ("The number of degrees in a circle is", "360"),
        ("The number of sides of a hexagon is", "six"),
        ("The number of days in a year is", "365"),
        ("The number of hours in a day is", "24"),
        ("The number of minutes in an hour is", "60"),
        ("The number of months in a year is", "twelve"),
        ("The binary representation of 10 is", "1010"),
    ],
    
    # Animals
    "animals": [
        ("The largest animal on Earth is the", "blue whale"),
        ("The fastest land animal is the", "cheetah"),
        ("The tallest animal is the", "giraffe"),
        ("The largest bird is the", "ostrich"),
        ("A baby dog is called a", "puppy"),
        ("A baby cat is called a", "kitten"),
        ("A group of lions is called a", "pride"),
        ("A group of fish is called a", "school"),
        ("The animal that produces honey is the", "bee"),
        ("The national animal of Australia is the", "kangaroo"),
    ],
    
    # Colors/properties
    "properties": [
        ("The color of the sky on a clear day is", "blue"),
        ("The color of grass is", "green"),
        ("The color of blood is", "red"),
        ("The color of snow is", "white"),
        ("The color of coal is", "black"),
        ("The hardest natural substance is", "diamond"),
        ("The most abundant gas in Earth's atmosphere is", "nitrogen"),
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
    p.add_argument('--knowledge-layer', type=int, default=33)
    args = p.parse_args()
    
    model, vocab_size = load_model(args.scaffold, args.lora, args.device)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    KL = args.knowledge_layer
    
    print(f"Mining knowledge subspace at layer L{KL}")
    print(f"Total categories: {len(KNOWLEDGE_DB)}")
    print(f"Total prompts: {sum(len(v) for v in KNOWLEDGE_DB.values())}")
    
    # ================================================================
    # STEP 1: Collect hidden states for all prompts
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Collecting hidden states")
    print("=" * 70)
    
    all_vectors = []
    all_labels = []
    all_categories = []
    all_answers = []
    
    for category, prompts in KNOWLEDGE_DB.items():
        print(f"\n  {category} ({len(prompts)} prompts):")
        for prompt, answer in prompts:
            ids = tok(prompt, return_tensors='pt').input_ids.to(args.device)
            vec = get_state_at_layer(model, ids, KL, args.device)
            
            # Also check what the model predicts
            with torch.no_grad():
                block_len = 64
                L = ids.shape[1]
                pad_len = (block_len - L % block_len) % block_len
                ids_padded = F.pad(ids, (pad_len, 0), value=0) if pad_len > 0 else ids
                logits = model(ids_padded)
                probs = F.softmax(logits[0, -1, :].float(), dim=-1)
                top1_id = probs.argmax().item()
                top1_tok = tok.decode([top1_id]).strip()
                
                ans_tokens = tok.encode(" " + answer, add_special_tokens=False)
                ans_prob = probs[ans_tokens[0]].item() if ans_tokens else 0
                ans_rank = (probs > ans_prob).sum().item() + 1
            
            all_vectors.append(vec[0])
            all_labels.append(f"{answer}")
            all_categories.append(category)
            all_answers.append(answer)
            
            print(f"    {answer:<20} top1={top1_tok:<12} P(ans)={ans_prob:.4f} rank={ans_rank}")
    
    V = torch.stack(all_vectors)  # (N, d_model)
    N = V.shape[0]
    
    # ================================================================
    # STEP 2: PCA on the full set
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 2: PCA of full knowledge space")
    print("=" * 70)
    
    V_centered = V - V.mean(dim=0)
    U, S, Vt = torch.svd(V_centered.float())
    
    print(f"\n  Variance explained:")
    total_var = (S ** 2).sum()
    cumul = 0
    for c in range(min(30, len(S))):
        var_pct = (S[c] ** 2 / total_var * 100).item()
        cumul += var_pct
        print(f"    PC{c:<4} {var_pct:>6.2f}%  cumulative: {cumul:>6.2f}%")
        if cumul > 99:
            break
    
    # ================================================================
    # STEP 3: Project and analyze clustering
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 3: Category clustering analysis")
    print("=" * 70)
    
    # Project onto top PCs
    projections = (V_centered.float() @ Vt[:, :10]).numpy()
    
    # Compute within-category and between-category similarities
    categories_unique = list(KNOWLEDGE_DB.keys())
    cat_indices = {cat: [i for i, c in enumerate(all_categories) if c == cat] 
                   for cat in categories_unique}
    
    V_norm = F.normalize(V.float(), dim=-1)
    sim_matrix = (V_norm @ V_norm.T).numpy()
    
    print(f"\n  {'Category':<15} {'Within-sim':>12} {'Between-sim':>12} {'Separation':>12}")
    print(f"  {'-'*53}")
    
    separations = []
    for cat in categories_unique:
        idx = cat_indices[cat]
        
        # Within-category similarity
        within_sims = []
        for i in idx:
            for j in idx:
                if i < j:
                    within_sims.append(sim_matrix[i, j])
        within_avg = np.mean(within_sims) if within_sims else 0
        
        # Between-category similarity (this cat vs all others)
        between_sims = []
        for i in idx:
            for j in range(N):
                if all_categories[j] != cat:
                    between_sims.append(sim_matrix[i, j])
        between_avg = np.mean(between_sims) if between_sims else 0
        
        separation = within_avg - between_avg
        separations.append(separation)
        print(f"  {cat:<15} {within_avg:>12.4f} {between_avg:>12.4f} {separation:>+12.4f}")
    
    avg_sep = np.mean(separations)
    print(f"\n  Average separation: {avg_sep:+.4f}")
    if avg_sep > 0.02:
        print(f"  → Categories form distinct clusters in the knowledge space")
    elif avg_sep > 0:
        print(f"  → Weak clustering — categories overlap significantly")
    else:
        print(f"  → No clustering — knowledge space is not category-organized")
    
    # ================================================================
    # STEP 4: Category centroid distances
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 4: Category centroid map")
    print("=" * 70)
    
    # Compute centroids in the 10D PC space
    centroids = {}
    for cat in categories_unique:
        idx = cat_indices[cat]
        centroid = projections[idx].mean(axis=0)
        centroids[cat] = centroid
    
    # Pairwise distances between centroids
    print(f"\n  Centroid distances (Euclidean in 10D PC space):")
    cats = list(centroids.keys())
    
    header = f"  {'':>15}" + "".join(f"{c[:8]:>10}" for c in cats)
    print(header)
    for i, c1 in enumerate(cats):
        row = f"  {c1:>15}"
        for j, c2 in enumerate(cats):
            dist = np.linalg.norm(centroids[c1] - centroids[c2])
            row += f" {dist:>9.1f}"
        print(row)
    
    # ================================================================
    # STEP 5: 2D map with labels
    # ================================================================
    print("\n\n" + "=" * 70)
    print("STEP 5: 2D knowledge map (PC1 vs PC2)")
    print("=" * 70)
    
    # Group by category, show centroid and spread
    for cat in categories_unique:
        idx = cat_indices[cat]
        pts = projections[idx]
        cx, cy = pts.mean(axis=0)[:2]
        spread = np.std(pts[:, :2])
        members = [all_labels[i] for i in idx]
        print(f"\n  {cat}:")
        print(f"    Centroid: ({cx:.1f}, {cy:.1f}), spread: {spread:.1f}")
        print(f"    Members: {', '.join(members[:8])}")
        
        # Individual points
        for k, i in enumerate(idx[:5]):
            print(f"      {all_labels[i]:<15} ({projections[i,0]:>7.1f}, {projections[i,1]:>7.1f})")
    
    # ================================================================
    # STEP 6: Export for visualization
    # ================================================================
    export = []
    for i in range(N):
        export.append({
            'label': all_labels[i],
            'category': all_categories[i],
            'pc1': float(projections[i, 0]),
            'pc2': float(projections[i, 1]),
            'pc3': float(projections[i, 2]) if projections.shape[1] > 2 else 0,
        })
    
    with open('knowledge_atlas_data.json', 'w') as f:
        json.dump(export, f, indent=2)
    print(f"\n\nExported {N} points to knowledge_atlas_data.json")
    print("Done.")


if __name__ == '__main__':
    main()
