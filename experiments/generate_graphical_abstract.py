"""
Generate graphical abstract for Ad Hoc Networks submission.
Spec: 531 × 1328 pixels (h × w), readable at 5 × 13 cm.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(13.28, 5.31), dpi=100)
fig.patch.set_facecolor('white')

ax.set_xlim(0, 13.28)
ax.set_ylim(0, 5.31)
ax.axis('off')

colors = {
    'input': '#E3F2FD',
    'conv': '#E8F5E9',
    'gru': '#FFF3E0',
    'fl': '#F3E5F5',
    'result': '#E0F7FA',
    'arrow': '#546E7A',
    'title': '#1A237E',
    'text': '#37474F',
}

def box(ax, x, y, w, h, color, text, subtext='', fc=None):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=fc or color, edgecolor='#37474F', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.08, text, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#1A237E')
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.22, subtext, ha='center', va='center',
                fontsize=5.5, color='#546E7A', style='italic')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5))

def bracket_arrow(ax, x, y1, y2, label):
    mid = (y1 + y2) / 2
    ax.plot([x, x], [y1, y2], color='#546E7A', lw=1.2)
    ax.plot([x-0.05, x], [y1, y1], color='#546E7A', lw=1.2)
    ax.plot([x-0.05, x], [y2, y2], color='#546E7A', lw=1.2)
    ax.text(x-0.15, mid, label, ha='right', va='center', fontsize=5.5, color='#546E7A')

# Title
ax.text(6.64, 5.0, 'Lightweight Federated IDS for IIoT', ha='center', va='center',
        fontsize=12, fontweight='bold', color=colors['title'])

ax.text(6.64, 4.7, 'CNN-GRU + Pruning + INT8 Quantization', ha='center', va='center',
        fontsize=8, color='#546E7A')

# Section 1: Architecture (left)
bx = 0.3
box_w = 3.8
box_h = 0.55

box(ax, bx, 3.8, box_w, box_h, colors['input'],
    'Input', '61 features, 200×3 windows', colors['input'])
arrow(ax, bx + box_w/2, 3.8, bx + box_w/2, 3.25)

box(ax, bx, 3.25, box_w, box_h, colors['conv'],
    '3× Depthwise Separable Conv1D', '32→64→128 filters, k=3', colors['conv'])
arrow(ax, bx + box_w/2, 3.25, bx + box_w/2, 2.7)

box(ax, bx, 2.7, box_w, box_h, colors['gru'],
    'GRU (64 hidden, 1 layer)', '3 gates vs 4 (LSTM)', colors['gru'])
arrow(ax, bx + box_w/2, 2.7, bx + box_w/2, 2.15)

box(ax, bx, 2.15, box_w, box_h, colors['result'],
    'FC → Classification', 'Normal / Attack (15 types)', colors['result'])

# Compression box
bx2 = 0.3
box(ax, bx2, 1.3, box_w, 0.5, colors['conv'],
    'Weight Pruning (τ=0.02)', '96.4% param reduction', '#E8F5E9')
arrow(ax, bx2 + box_w/2, 1.3, bx2 + box_w/2, 0.8)

box(ax, bx2, 0.8, box_w, 0.5, colors['gru'],
    'INT8 Quantization', '99.1% total size reduction (3.75 MB → 32 KB)', '#FFF3E0')

# Section 2: FL Diagram (center)
cx = 4.7
fl_w = 3.5
fl_h = 1.0

ax.text(cx + fl_w/2, 4.15, 'Federated Learning', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['title'])

server = FancyBboxPatch((cx + fl_w/2 - 0.6, 3.1), 1.2, 0.6,
                         boxstyle="round,pad=0.08", facecolor='#F3E5F5',
                         edgecolor='#6A1B9A', linewidth=1.5)
ax.add_patch(server)
ax.text(cx + fl_w/2, 3.4, 'Server\n(FedAvg)', ha='center', va='center',
        fontsize=6, fontweight='bold', color='#4A148C')

clients_y = [2.2, 1.6, 1.0, 0.4]
for i, cy in enumerate(clients_y):
    c = FancyBboxPatch((cx + i * 0.7 + 0.3, cy), 0.7, 0.4,
                        boxstyle="round,pad=0.05", facecolor='#E8EAF6',
                        edgecolor='#3949AB', linewidth=1)
    ax.add_patch(c)
    ax.text(cx + i * 0.7 + 0.65, cy + 0.2, f'Client {i+1}',
            ha='center', va='center', fontsize=4.5, color='#283593')
    arrow(ax, cx + i * 0.7 + 0.65, cy + 0.4, cx + fl_w/2, 3.1)

# Section 3: Results (right)
rx = 8.8
rw = 4.2

ax.text(rx + rw/2, 4.15, 'Key Results', ha='center', va='center',
        fontsize=9, fontweight='bold', color=colors['title'])

results = [
    ('Binary Accuracy', '99.55-100% (3 datasets)'),
    ('Multi-Class (15 types)', '95.47% (Edge-IIoTset)'),
    ('Model Size Reduction', '94.6% (3.75 MB → 0.20 MB)'),
    ('+ Quantized Size', '99.1% (3.75 MB → 32 KB)'),
    ('Inference Speedup', '3.1-3.5× vs baseline'),
    ('FL Communication', '95% overhead reduction'),
    ('FLOPs Reduction', '94.1% (7.94M → 0.47M)'),
]

for i, (label, value) in enumerate(results):
    y_pos = 3.6 - i * 0.48
    ax.text(rx + 0.2, y_pos, label, ha='left', va='center',
            fontsize=5.5, color='#37474F')
    ax.text(rx + rw - 0.2, y_pos, value, ha='right', va='center',
            fontsize=5.5, fontweight='bold', color='#1A237E')
    if i < len(results) - 1:
        ax.plot([rx + 0.2, rx + rw - 0.2], [y_pos - 0.2, y_pos - 0.2],
                color='#E0E0E0', lw=0.5)

# Dataset labels at bottom
ax.text(1.5, 0.25, 'Datasets: X-IIoTID  |  WUSTL-IIoT-2021  |  Edge-IIoTset',
        ha='center', va='center', fontsize=6, color='#78909C', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig('D:\\iiot-ids-research\\paper\\graphical_abstract.png',
            dpi=100, bbox_inches='tight', facecolor='white')
print(f"Saved: paper/graphical_abstract.png ({531}×{1328} px)")
