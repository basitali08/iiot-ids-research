"""
Regenerate all paper figures from real experimental results.
Output: results/figures/  and  paper/
"""

import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(RESULTS, 'figures')
PAPER_DIR = os.path.join(ROOT, 'paper')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RESULTS, 'real_data_results.json')) as f:
    data = json.load(f)

COLORS = ['#2E86AB', '#A23B72', '#F18F01']
MODEL_SHORT = ['CNN-BiLSTM', 'CNN-GRU', 'Pruned CNN-GRU']

dataset_order = ['WUSTL-IIoT', 'X-IIoTID', 'Edge-IIoTset']

def get_model_list(dataset):
    return list(data[dataset]['models'].keys())

def get_values(dataset, metric):
    models = get_model_list(dataset)
    return [data[dataset]['models'][m][metric] for m in models]

def create_efficiency_figure(dataset):
    models = get_model_list(dataset)
    sizes = get_values(dataset, 'model_size_mb')
    params = [v / 1000 for v in get_values(dataset, 'num_parameters')]
    times = get_values(dataset, 'inference_time_ms')
    flops = get_values(dataset, 'flops_millions')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        (sizes, 'Model Size (MB)', '(a) Size'),
        (params, 'Parameters (K)', '(b) Parameters'),
        (times, 'Inference Time (ms)', '(c) Speed'),
        (flops, 'FLOPs (M)', '(d) Computation'),
    ]
    for ax, (vals, ylabel, title) in zip(axes.flat, metrics):
        bars = ax.bar(MODEL_SHORT, vals, color=COLORS, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, vals):
            label = f'{val:.2f}' if val < 100 else f'{val:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25)

    plt.suptitle(f'{dataset} — Efficiency Comparison', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'efficiency_{dataset.lower().replace("-", "_").replace(" ", "_")}.png'
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    cp_path = os.path.join(PAPER_DIR, fname)
    import shutil; shutil.copy2(path, cp_path)
    print(f'  {fname}')

def create_main_comparison():
    """Comparison across all 3 datasets: 4 subplots (Params, Size, Time, FLOPs)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(dataset_order))
    width = 0.25

    metrics = [
        ('num_parameters', 'Parameters', lambda v: f'{v/1000:.0f}K'),
        ('model_size_mb', 'Model Size (MB)', lambda v: f'{v:.3f}'),
        ('inference_time_ms', 'Inference (ms)', lambda v: f'{v:.2f}'),
        ('flops_millions', 'FLOPs (M)', lambda v: f'{v:.2f}'),
    ]

    for ax, (key, ylabel, fmt) in zip(axes.flat, metrics):
        for i, ds in enumerate(dataset_order):
            models = get_model_list(ds)
            vals = [data[ds]['models'][m][key] / (1000 if key == 'num_parameters' else 1) for m in models]
            bars = ax.bar(x[i] + (i-1)*width - width, vals, width*0.8,
                         color=COLORS, edgecolor='black', linewidth=0.5, label=MODEL_SHORT[i] if i == 0 else '')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        fmt(val * (1000 if key == 'num_parameters' else 1)) if key != 'num_parameters' else f'{val:.0f}K',
                        ha='center', va='bottom', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=11)
        ds_short = [d.replace('IIoT', '').replace('-', ' ') for d in dataset_order]
        ax.set_xticks(x)
        ax.set_xticklabels(ds_short, fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    axes.flat[0].legend(MODEL_SHORT, fontsize=9, loc='upper right')
    plt.suptitle('Model Efficiency Comparison Across Datasets', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'efficiency_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    cp_path = os.path.join(PAPER_DIR, 'efficiency_comparison.png')
    import shutil; shutil.copy2(path, cp_path)
    print(f'  efficiency_comparison.png')

def create_improvement_chart():
    """% reduction + speedup vs baseline (first dataset = X-IIoTID as representative)."""
    ds = 'X-IIoTID'
    models = get_model_list(ds)
    b = data[ds]['models'][models[0]]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    categories = ['Parameters', 'Model Size', 'FLOPs', 'Inference Time']
    x = np.arange(len(categories))
    width = 0.35

    reductions = []
    speedups = []
    for i in [1, 2]:
        m = data[ds]['models'][models[i]]
        reductions.append([
            (1 - m['num_parameters'] / b['num_parameters']) * 100,
            (1 - m['model_size_mb'] / b['model_size_mb']) * 100,
            (1 - m['flops_millions'] / b['flops_millions']) * 100,
            0,
        ])
        speedups.append(b['inference_time_ms'] / m['inference_time_ms'])

    bars1 = ax1.bar(x - width/2, reductions[0][:3] + [0], width,
                    label='CNN-GRU (Lightweight)', color='#A23B72', edgecolor='black')
    bars2 = ax1.bar(x + width/2, reductions[1][:3] + [0], width,
                    label='Pruned CNN-GRU', color='#F18F01', edgecolor='black')

    for bars, vals in [(bars1, reductions[0][:3] + [0]), (bars2, reductions[1][:3] + [0])]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.bar(x[3] - width/2, speedups[0], width, color='#A23B72', alpha=0.5, edgecolor='black')
    ax2.bar(x[3] + width/2, speedups[1], width, color='#F18F01', alpha=0.5, edgecolor='black')
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_ylim(0, max(speedups) * 1.4)
    ax2.text(x[3] - width/2, speedups[0] + 0.15, f'{speedups[0]:.2f}×',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#A23B72')
    ax2.text(x[3] + width/2, speedups[1] + 0.15, f'{speedups[1]:.2f}×',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#F18F01')

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.set_ylabel('Reduction (%)', fontsize=12)
    ax1.set_title(f'Efficiency Improvement vs Baseline ({ds})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 110)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'improvement_chart.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    cp_path = os.path.join(PAPER_DIR, 'improvement_chart.png')
    import shutil; shutil.copy2(path, cp_path)
    print(f'  improvement_chart.png')


if __name__ == '__main__':
    print('Generating figures from real data results...')
    print('\nPer-dataset efficiency:')
    for ds in dataset_order:
        if ds in data:
            create_efficiency_figure(ds)
    print('\nMain figures:')
    create_main_comparison()
    create_improvement_chart()
    print(f'\nAll figures saved to {FIG_DIR}')
    print(f'Copied to {PAPER_DIR}')
