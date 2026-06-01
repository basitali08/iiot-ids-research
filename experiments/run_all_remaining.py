import sys, os, json, copy, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.metrics import (
    evaluate_model, compute_model_size_mb, count_parameters,
    measure_inference_time, estimate_flops, train_one_epoch, federated_averaging
)
from utils.data_loader import create_federated_clients

device = 'cpu'
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')
fig_dir = os.path.join(results_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)


def load_csv_generic(filepath, label_col=None, n_rows=20000):
    df = pd.read_csv(filepath, low_memory=False, nrows=n_rows)
    if label_col and label_col in df.columns:
        lc = label_col
    else:
        lc = df.columns[-1]
    df['label'] = df[lc].apply(
        lambda x: 0 if str(x).strip().lower() in ['0', 'normal', 'benign', ''] else 1
    )
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['label', lc]]
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [lc]]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    feature_cols = num_cols + cat_cols
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, X.shape[1]


def train_one_dataset(name, X, y, epochs=10, bsize=128):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Attack ratio: {y.mean():.2%}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    input_dim = X.shape[1]
    configs = [
        ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, 2)),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, 2)),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, 2, pruning_threshold=0.02)),
    ]
    results = {}
    for mname, model in configs:
        t0 = time.time()
        model.train()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for ep in range(epochs):
            loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, bsize, device)
            if (ep + 1) % 5 == 0:
                vm = evaluate_model(model, X_val, y_val, device)
                print(f"    {mname[:20]:20} Epoch {ep+1}/{epochs} - Loss: {loss:.4f}, Val Acc: {vm['accuracy']:.4f}")
        me = evaluate_model(model, X_test, y_test, device)
        sz = compute_model_size_mb(model)
        pa = count_parameters(model)
        ti = measure_inference_time(model, X_test, device)
        fl = estimate_flops(model, input_dim, device)
        sp = 0.0
        if hasattr(model, 'apply_pruning'):
            model.apply_pruning()
            sp = model.get_sparsity()
        elapsed = time.time() - t0
        print(f"    {mname[:20]:20} Acc={me['accuracy']:.4f} F1={me['f1_score']:.4f} Size={sz:.3f}MB Params={pa:,} Inf={ti:.3f}ms FLOPs={fl:.2f}M Sparsity={sp:.2%} [{elapsed:.0f}s]")
        results[mname] = {
            'accuracy': float(me['accuracy']),
            'precision': float(me['precision']),
            'recall': float(me['recall']),
            'f1_score': float(me['f1_score']),
            'auc': float(me['auc']),
            'model_size_mb': float(sz),
            'num_parameters': int(pa),
            'inference_time_ms': float(ti),
            'flops_millions': float(fl),
            'sparsity': float(sp),
        }
    return {name: {'input_dim': input_dim, 'models': results}}


def run_fl_on_real(X, y, dataset_name, num_clients=5, rounds=12, local_epochs=2):
    print(f"\n{'='*60}")
    print(f"  FEDERATED LEARNING - {dataset_name}")
    print(f"{'='*60}")
    input_dim = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {input_dim}")
    all_results = {}
    for non_iid, mode_name in [(False, 'IID'), (True, 'Non-IID')]:
        client_data = create_federated_clients(
            X_train, y_train, num_clients=num_clients, non_iid=non_iid
        )
        for model_name, ModelClass in [
            ('CNN-BiLSTM', CNNBiLSTM),
            ('CNN-GRU (Lightweight)', LightweightCNNGRU),
        ]:
            key = f"{model_name}_{mode_name}"
            print(f"\n    --- {key} ---")
            global_model = ModelClass(input_dim, 2)
            client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
            criterion = nn.CrossEntropyLoss()
            round_accs = []
            for rnd in range(rounds):
                for ci, (Xc, yc) in enumerate(client_data):
                    cm = client_models[ci]
                    opt = torch.optim.Adam(cm.parameters(), lr=0.001)
                    for _ in range(local_epochs):
                        train_one_epoch(cm, Xc, yc, opt, criterion, 64, device)
                global_model = federated_averaging(global_model, client_models)
                for cm in client_models:
                    cm.load_state_dict(global_model.state_dict())
                m = evaluate_model(global_model, X_test, y_test, device)
                round_accs.append(float(m['accuracy']))
                if (rnd + 1) % 4 == 0:
                    print(f"      Round {rnd+1}/{rounds}: Acc={m['accuracy']:.4f}")
            print(f"    Final: {key:<30} Acc={round_accs[-1]:.4f}")
            all_results[key] = round_accs
    return all_results


def generate_figures():
    with open(os.path.join(results_dir, 'real_data_results.json')) as f:
        data = json.load(f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for dname, dres in data.items():
        models = list(dres['models'].keys())
        sizes = [dres['models'][m]['model_size_mb'] for m in models]
        params = [dres['models'][m]['num_parameters'] / 1000 for m in models]
        times = [dres['models'][m]['inference_time_ms'] for m in models]
        flops = [dres['models'][m]['flops_millions'] for m in models]
        accs = [dres['models'][m]['accuracy'] for m in models]
        f1s = [dres['models'][m]['f1_score'] for m in models]

        colors = ['#2E86AB', '#A23B72', '#F18F01']
        labels_short = [m.split(' (')[0] for m in models]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # (a) Model Size
        ax = axes[0, 0]
        bars = ax.bar(labels_short, sizes, color=colors, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Model Size (MB)', fontsize=12)
        ax.set_title(f'({dname}) Model Size', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(sizes) * 1.2)

        # (b) Parameters
        ax = axes[0, 1]
        bars = ax.bar(labels_short, params, color=colors, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.0f}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Parameters (thousands)', fontsize=12)
        ax.set_title(f'({dname}) Model Parameters', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(params) * 1.2)

        # (c) Inference Time
        ax = axes[0, 2]
        bars = ax.bar(labels_short, times, color=colors, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Inference Time (ms/sample)', fontsize=12)
        ax.set_title(f'({dname}) Inference Speed', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(times) * 1.2)

        # (d) FLOPs
        ax = axes[1, 0]
        bars = ax.bar(labels_short, flops, color=colors, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, flops):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('FLOPs (millions)', fontsize=12)
        ax.set_title(f'({dname}) Computational Cost', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(flops) * 1.2)

        # (e) Detection Performance
        ax = axes[1, 1]
        x = np.arange(len(labels_short))
        width = 0.35
        ax.bar(x - width/2, accs, width, label='Accuracy', color='steelblue', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, f1s, width, label='F1-Score', color='lightcoral', edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'({dname}) Detection Performance', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0.95, 1.01)
        ax.grid(axis='y', alpha=0.3)

        # (f) Summary Table
        ax = axes[1, 2]
        ax.axis('off')
        b = list(dres['models'].values())[0]
        cell_text = [
            ['Params', f'{b["num_parameters"]:,}', f'{list(dres["models"].values())[1]["num_parameters"]:,}',
             f'{list(dres["models"].values())[2]["num_parameters"]:,}'],
            ['Size (MB)', f'{b["model_size_mb"]:.3f}', f'{list(dres["models"].values())[1]["model_size_mb"]:.3f}',
             f'{list(dres["models"].values())[2]["model_size_mb"]:.3f}'],
            ['Inf. (ms)', f'{b["inference_time_ms"]:.3f}', f'{list(dres["models"].values())[1]["inference_time_ms"]:.3f}',
             f'{list(dres["models"].values())[2]["inference_time_ms"]:.3f}'],
            ['FLOPs (M)', f'{b["flops_millions"]:.2f}', f'{list(dres["models"].values())[1]["flops_millions"]:.2f}',
             f'{list(dres["models"].values())[2]["flops_millions"]:.2f}']]
        table = ax.table(cellText=cell_text, colLabels=['Metric', 'Baseline', 'Lightweight', 'Pruned'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(color='white', fontweight='bold')

        plt.suptitle(f'{dname}: Efficiency vs. Performance Comparison', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        fname = f'efficiency_{dname.lower().replace("-","_").replace(" ","_")}.png'
        fig_path = os.path.join(fig_dir, fname)
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  Figure saved: {fname}')

    # Combined summary figure with improvement chart
    w = list(data.values())[0]
    b = list(w['models'].values())[0]
    models_list = list(w['models'].keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Parameters', 'Model Size', 'FLOPs', 'Inference Time']
    reduction = []
    for i in [1, 2]:
        p_red = (1 - list(w['models'].values())[i]['num_parameters'] / b['num_parameters']) * 100
        s_red = (1 - list(w['models'].values())[i]['model_size_mb'] / b['model_size_mb']) * 100
        f_red = (1 - list(w['models'].values())[i]['flops_millions'] / b['flops_millions']) * 100
        t_speedup = b['inference_time_ms'] / list(w['models'].values())[i]['inference_time_ms']
        reduction.append([p_red, s_red, f_red, t_speedup])
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, reduction[0][:3] + [0], width, label='CNN-GRU (Lightweight)', color='#A23B72', edgecolor='black')
    bars2 = ax.bar(x + width/2, reduction[1][:3] + [0], width, label='Pruned CNN-GRU', color='#F18F01', edgecolor='black')
    for bars, vals in [(bars1, reduction[0][:3] + [0]), (bars2, reduction[1][:3] + [0])]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2 = ax.twinx()
    sp1, sp2 = reduction[0][3], reduction[1][3]
    ax2.bar(x[3], sp1, width, color='#A23B72', alpha=0.3, edgecolor='black')
    ax2.bar(x[3] + width, sp2, width, color='#F18F01', alpha=0.3, edgecolor='black')
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_ylim(0, max(sp1, sp2) * 1.3)
    ax2.text(x[3], sp1 + 0.2, f'{sp1:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#A23B72')
    ax2.text(x[3] + width, sp2 + 0.2, f'{sp2:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#F18F01')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('Reduction (%)', fontsize=12)
    ax.set_title('Efficiency Improvements vs. CNN-BiLSTM Baseline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'improvement_chart.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: improvement_chart.png')
    print('All figures generated.')


if __name__ == '__main__':
    results_file = os.path.join(results_dir, 'real_data_results.json')
    with open(results_file) as f:
        all_results = json.load(f)

    # === 1. Edge-IIoTset ===
    edge_path = os.path.join(base_dir, 'data', 'Edge-IIoTset', 'edge_iiot_binary.csv')
    if 'Edge-IIoTset' not in all_results and os.path.exists(edge_path):
        print("\n" + "="*60)
        print("PHASE 1: Edge-IIoTset")
        print("="*60)
        X, y, _ = load_csv_generic(edge_path, n_rows=20000)
        edge_res = train_one_dataset('Edge-IIoTset', X, y, epochs=10)
        all_results.update(edge_res)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print("Edge-IIoTset results saved.")
    else:
        print("Edge-IIoTset already processed or not found. Skipping.")

    # === 2. Federated Learning on real data ===
    fl_file = os.path.join(results_dir, 'fl_real_results.json')
    if not os.path.exists(fl_file):
        print("\n" + "="*60)
        print("PHASE 2: FL on Real Data")
        print("="*60)
        wustl_path = os.path.join(base_dir, 'data', 'WUSTL-IIoT', 'wustl_iiot_2021.csv')
        print("Loading WUSTL-IIoT for FL...")
        X, y, _ = load_csv_generic(wustl_path, n_rows=5000)
        fl_results = run_fl_on_real(X, y, 'WUSTL-IIoT', num_clients=5, rounds=12, local_epochs=2)
        with open(fl_file, 'w') as f:
            json.dump(fl_results, f, indent=2)
        print("FL results saved.")
    else:
        print("FL real results already exist. Skipping.")

    # === 3. Generate Figures ===
    print("\n" + "="*60)
    print("PHASE 3: Generate Figures")
    print("="*60)
    generate_figures()

    # === 4. Print Summary ===
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    with open(results_file) as f:
        ar = json.load(f)
    for dn, res in ar.items():
        print(f"\n  --- {dn} ---")
        b = list(res['models'].values())[0]
        for mn, m in res['models'].items():
            short = mn.split(' (')[0]
            sz_red = (1 - m['model_size_mb']/b['model_size_mb'])*100
            sp_up = b['inference_time_ms']/m['inference_time_ms']
            print(f"    {short:<20} Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} "
                  f"Size={m['model_size_mb']:.3f}MB Params={m['num_parameters']:,} "
                  f"Inf={m['inference_time_ms']:.2f}ms FLOPs={m['flops_millions']:.2f}M"
                  + (f" Sparsity={m['sparsity']:.1%}" if m['sparsity'] > 0 else "")
                  + (f" ({sz_red:+.1f}%, {sp_up:.2f}x)" if 'Baseline' not in mn else ""))

    if os.path.exists(fl_file):
        with open(fl_file) as f:
            flr = json.load(f)
        print("\n  --- FL Real Data ---")
        for key, accs in flr.items():
            print(f"    {key:<35} Final Acc: {accs[-1]:.4f}, Converged at: round {next((i+1 for i,a in enumerate(accs) if a >= 0.98), 'N/A')}")

    print(f"\nAll done! Results: {results_file}, Figures: {fig_dir}")
