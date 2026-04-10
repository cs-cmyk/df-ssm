#!/usr/bin/env python3
"""
Render 4-panel knowledge atlas from atlas_large.json.

Usage:
    python render_atlas.py atlas_large.json
    python render_atlas.py atlas_large.json --output atlas.png --dpi 300
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse


COLORS = {
    'capitals':   '#534AB7',
    'languages':  '#D85A30',
    'continents': '#0F6E56',
    'elements':   '#185FA5',
    'physics':    '#D4537E',
    'scientists': '#639922',
    'writers':    '#BA7517',
    'companies':  '#E24B4A',
    'animals':    '#888780',
    'colors':     '#993556',
    'currencies': '#854F0B',
    'math':       '#3B6D11',
    'food':       '#A32D2D',
    'music':      '#3C3489',
    'sports':     '#085041',
    'medical':    '#712B13',
    'history':    '#0C447C',
    'mythology':  '#72243E',
    'materials':  '#5F5E5A',
}

SPACE_NAMES = {
    'L2':  'Intent space',
    'L14': 'Translation space',
    'L32': 'Knowledge space',
    'L46': 'Output space',
}


def draw_panel(ax, layer_data, layer_name):
    pts = layer_data['points']
    sep = layer_data['avg_sep']
    var_pct = layer_data.get('var_10pc', 0)
    space = SPACE_NAMES.get(layer_name, layer_name)
    
    # Group by category
    cats = {}
    for p in pts:
        cats.setdefault(p['category'], []).append((p['x'], p['y']))
    
    # Draw points and ellipses
    for cat, points in cats.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = COLORS.get(cat, '#888888')
        
        # Scatter
        ax.scatter(xs, ys, c=color, s=8, alpha=0.6, edgecolors='none', zorder=2)
        
        # Centroid
        cx, cy = np.mean(xs), np.mean(ys)
        
        # Ellipse (2-sigma)
        if len(xs) > 2:
            std_x = np.std(xs)
            std_y = np.std(ys)
            if std_x > 0 and std_y > 0:
                ellipse = Ellipse(
                    (cx, cy), width=std_x * 3, height=std_y * 3,
                    fill=True, facecolor=color, alpha=0.08,
                    edgecolor=color, linewidth=0.5, linestyle='-',
                    zorder=1
                )
                ax.add_patch(ellipse)
        
        # Label at centroid
        ax.annotate(
            cat, (cx, cy),
            fontsize=6, color=color, fontweight='500',
            ha='center', va='center', alpha=0.9,
            zorder=3,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                      edgecolor='none', alpha=0.7)
        )
    
    ax.set_title(f'{space}\n(avg separation: {sep:+.3f})',
                 fontsize=11, fontweight='500', pad=8)
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(0, color='#00000010', linewidth=0.5, zorder=0)
    ax.axvline(0, color='#00000010', linewidth=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('json_file', help='Path to atlas_large.json')
    p.add_argument('--output', '-o', default='atlas_4panel.png')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()
    
    with open(args.json_file) as f:
        data = json.load(f)
    
    layers = ['L2', 'L14', 'L32', 'L46']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Knowledge atlas: 445 prompts across 19 categories',
                 fontsize=14, fontweight='500', y=0.98)
    
    for idx, layer in enumerate(layers):
        ax = axes[idx // 2][idx % 2]
        draw_panel(ax, data[layer], layer)
    
    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[c],
                    markersize=6, label=c)
        for c in sorted(COLORS.keys())
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=7,
        fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.0, handletextpad=0.3
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved to {args.output} ({args.dpi} dpi)')
    plt.show()


if __name__ == '__main__':
    main()
