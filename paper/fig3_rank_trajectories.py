#!/usr/bin/env python3
"""Figure 3: Answer rank trajectories through the network (logit lens)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

layers = ['emb','L0','L1','L2','L3','L7','L11','L15','L19','L23','L27','L29','L31','L33','L35','L39','L43','L46','nf']
layer_nums = [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 30, 32, 34, 36, 40, 44, 47, 49]

data = {
    'Paris':     {'ranks': [1639,27519,27593,30541,26942,34787,39115,5055,509,158,63,11,3,4,5,5,2,4,4],
                  'color': '#534AB7', 'style': '-'},
    'Berlin':    {'ranks': [4726,42391,44547,43058,37533,44198,39401,4881,430,178,52,8,3,4,6,5,3,2,2],
                  'color': '#1D9E75', 'style': '-'},
    'French':    {'ranks': [1795,29013,41045,43147,33861,40652,30668,6147,825,59,20,6,3,7,7,2,2,2,2],
                  'color': '#D85A30', 'style': '-'},
    'H (water)': {'ranks': [109,30602,37311,41984,28132,33858,30093,3515,240,127,26,17,7,4,3,2,7,2,3],
                  'color': '#185FA5', 'style': '-'},
    'four (2+2)':{'ranks': [48550,45415,45944,44631,42027,32420,5772,1077,40,21,20,5,7,4,5,3,2,2,2],
                  'color': '#BA7517', 'style': '-'},
    'Microsoft': {'ranks': [3942,38165,30447,31984,22971,26230,30276,5345,156,62,20,7,4,3,3,2,5,3,4],
                  'color': '#E24B4A', 'style': '-'},
    'green':     {'ranks': [1640,6701,8591,13116,16623,13840,18514,5550,91,27,12,22,18,12,4,3,2,2,2],
                  'color': '#639922', 'style': '-'},
}

fig, ax = plt.subplots(figsize=(12, 5.5))

# Phase backgrounds
ax.axvspan(0, 12, alpha=0.04, color='#534AB7', zorder=0)
ax.axvspan(14, 36, alpha=0.04, color='#1D9E75', zorder=0)
ax.axvspan(36, 49, alpha=0.04, color='#D85A30', zorder=0)

ax.text(6, 0.15, 'Noise', ha='center', fontsize=10, color='#534AB7', alpha=0.6, fontweight='500')
ax.text(25, 0.15, 'Ascent', ha='center', fontsize=10, color='#1D9E75', alpha=0.6, fontweight='500')
ax.text(43, 0.15, 'Plateau', ha='center', fontsize=10, color='#D85A30', alpha=0.6, fontweight='500')

for name, d in data.items():
    log_ranks = [np.log10(max(r, 1)) for r in d['ranks']]
    ax.plot(layer_nums, log_ranks, color=d['color'], linewidth=2, alpha=0.85,
            marker='o', markersize=3.5, label=name, linestyle=d['style'])

# Y axis: log scale with readable labels
yticks = [0, 0.7, 1, 1.7, 2, 2.7, 3, 3.7, 4, 4.7]
ylabels = ['1', '5', '10', '50', '100', '500', '1K', '5K', '10K', '50K']
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_ylim(4.9, -0.1)  # inverted: rank 1 at top

# X axis
xtick_pos = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 49]
xtick_labels = ['emb', 'L3', 'L7', 'L11', 'L15', 'L19', 'L23', 'L27', 'L31', 'L35', 'L39', 'L43', 'nf']
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_labels, fontsize=9)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Answer rank (log scale, lower = better)', fontsize=12)
ax.set_title('Answer rank trajectories through the network', fontsize=13, fontweight='500')

ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Rank 1 line
ax.axhline(y=0, color='#00000020', linewidth=0.5, linestyle='-')
ax.text(49.5, 0, 'rank 1', fontsize=8, color='#888780', va='center')

plt.tight_layout()
plt.savefig('fig3_rank_trajectories.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fig3_rank_trajectories.pdf', bbox_inches='tight', facecolor='white')
print('Saved fig3_rank_trajectories.png/pdf')
