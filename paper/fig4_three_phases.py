#!/usr/bin/env python3
"""Figure 4: Three-phase processing model — clustering evolution across layers."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

# Data from knowledge_crystallization.py output
layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
layer_labels = ['emb','L1','L3','L5','L7','L9','L11','L13','L15','L17','L19','L21','L23','L25','L27','L29','L31','L33','L35','L37','L39','L41','L43','L45','L47']

# Category separations from crystallization output
cats_data = {
    'capitals':  [0.164,0.128,0.351,0.273,0.237,0.253,0.268,0.294,0.265,0.225,0.239,0.235,0.221,0.271,0.292,0.283,0.272,0.234,0.226,0.218,0.189,0.150,0.128,0.095,0.114],
    'languages': [0.154,0.121,0.328,0.262,0.233,0.270,0.284,0.295,0.271,0.232,0.247,0.251,0.239,0.287,0.311,0.277,0.282,0.231,0.212,0.215,0.187,0.148,0.119,0.087,0.106],
    'elements':  [0.154,0.112,0.310,0.247,0.215,0.226,0.248,0.276,0.253,0.214,0.237,0.240,0.232,0.246,0.267,0.268,0.254,0.241,0.232,0.221,0.188,0.151,0.130,0.096,0.113],
    'writers':   [0.154,0.106,0.344,0.244,0.209,0.226,0.248,0.277,0.258,0.223,0.255,0.259,0.248,0.289,0.308,0.319,0.292,0.263,0.257,0.240,0.204,0.163,0.141,0.103,0.125],
    'animals':   [0.494,0.336,0.417,0.319,0.301,0.288,0.300,0.257,0.230,0.183,0.189,0.184,0.168,0.193,0.200,0.214,0.203,0.195,0.189,0.165,0.153,0.119,0.101,0.076,0.098],
    'colors':    [-0.024,-0.012,0.111,0.071,0.064,0.075,0.102,0.142,0.130,0.107,0.119,0.118,0.112,0.145,0.162,0.167,0.146,0.135,0.134,0.133,0.119,0.091,0.076,0.056,0.070],
}

avg_sep = []
for i in range(len(layers)):
    vals = [cats_data[c][i] for c in cats_data]
    avg_sep.append(np.mean(vals))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], sharex=True)

# --- Top panel: per-category separation ---
colors = {
    'capitals': '#534AB7', 'languages': '#D85A30', 'elements': '#185FA5',
    'writers': '#BA7517', 'animals': '#888780', 'colors': '#993556'
}

for cat, seps in cats_data.items():
    ax1.plot(layers, seps, color=colors[cat], linewidth=1.5, alpha=0.7, label=cat)

ax1.plot(layers, avg_sep, color='#2C2C2A', linewidth=2.5, alpha=0.9, label='average', linestyle='-')

# Phase backgrounds
for ax in [ax1, ax2]:
    ax.axvspan(0, 4, alpha=0.06, color='#534AB7', zorder=0)
    ax.axvspan(24, 36, alpha=0.06, color='#1D9E75', zorder=0)
    ax.axvspan(36, 48, alpha=0.06, color='#D85A30', zorder=0)

ax1.text(2, -0.03, 'Categorize', ha='center', fontsize=11, color='#534AB7', fontweight='500', alpha=0.7)
ax1.text(30, -0.03, 'Recall', ha='center', fontsize=11, color='#1D9E75', fontweight='500', alpha=0.7)
ax1.text(42, -0.03, 'Format', ha='center', fontsize=11, color='#D85A30', fontweight='500', alpha=0.7)

ax1.set_ylabel('Category separation\n(within - between similarity)', fontsize=11)
ax1.set_title('Three-phase knowledge processing', fontsize=13, fontweight='500')
ax1.legend(fontsize=9, loc='upper right', ncol=4, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(0, color='#00000015', linewidth=0.5)
ax1.set_ylim(-0.05, 0.55)

# --- Bottom panel: within-category similarity (convergence in format phase) ---
# From crystallization output
within_sims = {
    'capitals':  [1.000,0.992,0.973,0.976,0.976,0.983,0.978,0.987,0.984,0.990,0.989,0.989,0.989,0.985,0.983,0.962,0.966,0.957,0.964,0.969,0.967,0.978,0.976,0.984,0.979],
    'writers':   [1.000,0.976,0.911,0.899,0.899,0.884,0.884,0.906,0.906,0.928,0.928,0.930,0.930,0.935,0.935,0.951,0.951,0.969,0.969,0.972,0.972,0.983,0.983,0.989,0.989],
}

for cat in ['capitals', 'writers']:
    ax2.plot(layers, within_sims[cat], color=colors[cat], linewidth=2, label=f'{cat} (within-sim)')

ax2.set_ylabel('Within-category\ncosine similarity', fontsize=11)
ax2.set_xlabel('Layer', fontsize=12)
ax2.legend(fontsize=9, loc='lower right', frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylim(0.88, 1.005)

ax2.set_xticks(layers[::2])
ax2.set_xticklabels(layer_labels[::2], fontsize=8, rotation=45)

ax2.annotate('Formatting convergence →\nall prompts become similar',
             xy=(42, 0.985), fontsize=9, color='#D85A30', alpha=0.7,
             ha='center')

plt.tight_layout()
plt.savefig('fig4_three_phases.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fig4_three_phases.pdf', bbox_inches='tight', facecolor='white')
print('Saved fig4_three_phases.png/pdf')
