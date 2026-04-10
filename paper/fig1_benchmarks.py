#!/usr/bin/env python3
"""Figure 1: GPU/CPU benchmarks and model size comparison."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Panel A: GPU throughput ---
ax = axes[0]
batches = ['Batch 1', 'Batch 8', 'Batch 32']
fp16 = [14, 116, 482]
binary = [299, 647, 1963]
speedups = ['21.4×', '5.6×', '4.1×']

x = range(len(batches))
w = 0.35
bars1 = ax.bar([i - w/2 for i in x], fp16, w, color='#D3D1C7', label='FP16 mamba-ssm')
bars2 = ax.bar([i + w/2 for i in x], binary, w, color='#534AB7', label='DF-SSM (1-bit)')

for i, (b, s) in enumerate(zip(bars2, speedups)):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 40, s,
            ha='center', va='bottom', fontsize=10, fontweight='500', color='#534AB7')

ax.set_ylabel('Tokens/sec')
ax.set_title('GPU throughput (A100)', fontweight='500')
ax.set_xticks(x)
ax.set_xticklabels(batches)
ax.legend(fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 2400)

# --- Panel B: CPU throughput ---
ax = axes[1]
methods = ['FP16\nPyTorch', 'DF-SSM\nAVX-512']
toks = [12, 22]
colors = ['#D3D1C7', '#534AB7']

bars = ax.bar(methods, toks, color=colors, width=0.5)
ax.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + 0.5, '1.8×',
        ha='center', va='bottom', fontsize=10, fontweight='500', color='#534AB7')

ax.set_ylabel('Tokens/sec')
ax.set_title('CPU throughput (Xeon, 4 threads)', fontweight='500')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 30)

# --- Panel C: Model size ---
ax = axes[2]
components = ['Scaffold\n(1-bit)', 'Embedding\n(int8)', 'LoRA\n(int8)', 'Other']
sizes = [155, 103, 12, 8]
colors_c = ['#534AB7', '#7F77DD', '#AFA9EC', '#D3D1C7']

bars = ax.bar(components, sizes, color=colors_c, width=0.6)
for b, s in zip(bars, sizes):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f'{s} MB',
            ha='center', va='bottom', fontsize=9)

ax.axhline(y=2688/10, color='#E24B4A', linewidth=1, linestyle='--', alpha=0.5)
ax.text(3.4, 2688/10 + 5, 'FP16 teacher: 2,688 MB', fontsize=8,
        color='#E24B4A', ha='right', alpha=0.7)

ax.set_ylabel('Size (MB)')
ax.set_title('Model composition (278 MB total)', fontweight='500')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 300)

plt.tight_layout()
plt.savefig('fig1_benchmarks.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fig1_benchmarks.pdf', bbox_inches='tight', facecolor='white')
print('Saved fig1_benchmarks.png/pdf')
