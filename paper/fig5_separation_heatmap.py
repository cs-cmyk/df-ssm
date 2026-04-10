#!/usr/bin/env python3
"""Figure 5: Category separation heatmap — 19 categories × 4 representational spaces."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
import argparse
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', default='atlas_large.json', help='Path to atlas_large.json')
    args = p.parse_args()

    # Data from atlas_large.py output (hardcoded as fallback)
    layer_names = ['L2\nIntent', 'L14\nTranslation', 'L32\nKnowledge', 'L46\nOutput']

    seps = {
        'continents':  [0.554, 0.498, 0.336, 0.084],
        'currencies':  [0.539, 0.394, 0.318, 0.102],
        'capitals':    [0.269, 0.340, 0.281, 0.090],
        'languages':   [0.264, 0.347, 0.295, 0.090],
        'animals':     [0.330, 0.257, 0.200, 0.069],
        'elements':    [0.246, 0.327, 0.301, 0.102],
        'writers':     [0.237, 0.337, 0.302, 0.099],
        'colors':      [0.215, 0.318, 0.278, 0.100],
        'companies':   [0.219, 0.212, 0.182, 0.064],
        'scientists':  [0.155, 0.159, 0.192, 0.066],
        'math':        [0.030, 0.118, 0.137, 0.069],
        'food':        [0.111, 0.148, 0.126, 0.037],
        'mythology':   [0.096, 0.097, 0.094, 0.035],
        'materials':   [0.046, 0.066, 0.108, 0.053],
        'sports':      [0.049, 0.033, 0.064, 0.038],
        'history':     [0.003, 0.044, 0.059, 0.018],
        'music':       [0.002, 0.031, 0.060, 0.031],
        'physics':     [-0.066, 0.004, 0.026, 0.021],
        'medical':     [-0.037, 0.003, 0.027, 0.011],
    }

    # Sort by L2 separation (highest first)
    cats_sorted = sorted(seps.keys(), key=lambda c: -seps[c][0])
    
    matrix = np.array([seps[c] for c in cats_sorted])
    n_cats = len(cats_sorted)
    n_layers = 4

    fig, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(12, 7),
                                           gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})

    # --- Heatmap ---
    im = ax_heat.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.55)

    ax_heat.set_xticks(range(n_layers))
    ax_heat.set_xticklabels(layer_names, fontsize=11)
    ax_heat.set_yticks(range(n_cats))
    ax_heat.set_yticklabels(cats_sorted, fontsize=10)

    # Annotate cells
    for i in range(n_cats):
        for j in range(n_layers):
            val = matrix[i, j]
            color = 'white' if val > 0.35 or val < -0.03 else '#2C2C2A'
            text = f'{val:+.3f}' if val != 0 else '0.000'
            ax_heat.text(j, i, text, ha='center', va='center', fontsize=8, color=color)

    ax_heat.set_title('Category separation by representational space', fontsize=13, fontweight='500')

    # Tier brackets
    ax_heat.axhline(1.5, color='white', linewidth=2)
    ax_heat.axhline(7.5, color='white', linewidth=2)

    ax_heat.text(-1.1, 0.75, 'Tier 1\nTemplate', ha='center', va='center',
                 fontsize=8, color='#0F6E56', fontweight='500', rotation=0)
    ax_heat.text(-1.1, 4.5, 'Tier 2\nSemantic', ha='center', va='center',
                 fontsize=8, color='#534AB7', fontweight='500', rotation=0)
    ax_heat.text(-1.1, 14, 'Tier 3\nMixed', ha='center', va='center',
                 fontsize=8, color='#A32D2D', fontweight='500', rotation=0)

    # --- Bar chart: average across layers ---
    avg_per_cat = [np.mean(seps[c]) for c in cats_sorted]
    bar_colors = []
    for i, c in enumerate(cats_sorted):
        if i < 2:
            bar_colors.append('#0F6E56')
        elif i < 8:
            bar_colors.append('#534AB7')
        else:
            bar_colors.append('#A32D2D')

    ax_bar.barh(range(n_cats), avg_per_cat, color=bar_colors, alpha=0.7, height=0.7)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel('Avg separation', fontsize=10)
    ax_bar.set_title('Average', fontsize=11, fontweight='500')
    ax_bar.axvline(0, color='#00000020', linewidth=0.5)
    ax_bar.set_xlim(-0.05, 0.45)
    ax_bar.invert_yaxis()
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal', fraction=0.04, pad=0.12,
                        label='Within − between cosine similarity')

    plt.savefig('fig5_separation_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig5_separation_heatmap.pdf', bbox_inches='tight', facecolor='white')
    print('Saved fig5_separation_heatmap.png/pdf')


if __name__ == '__main__':
    main()
