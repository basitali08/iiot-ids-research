import sys, os, json
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
    measure_inference_time, estimate_flops, train_one_epoch
)

device = 'cpu'


def load_dataset(filepath, label_col_name=None, sample_rows=None):
    name = os.path.basename(os.path.dirname(filepath))
    print(f"Loading {name}...")
    nrows = sample_rows
    df = pd.read_csv(filepath, low_memory=False, nrows=nrows)
    print(f"  Shape: {df.shape}")

    if label_col_name and label_col_name in df.columns:
        label_col = label_col_name
    else:
        label_col = df.columns[-1]

    df['label'] = df[label_col].apply(
        lambda x: 0 if str(x).strip().lower() in ['0', 'normal', 'benign', ''] else 1
    )

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['label', label_col]]

    cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [label_col]]

    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"  Final: {X.shape} ({X.shape[1]} features), Attack ratio: {y.mean():.2%}")
    return X, y


def run_on_dataset(name, load_fn, filepath, label_col=None, n_rows=30000):
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"{'='*70}")

    X, y = load_fn(filepath, label_col_name=label_col, sample_rows=n_rows)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_dim = X.shape[1]
    results = {'dataset': name, 'input_dim': input_dim, 'models': {}}

    models_config = [
        ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, num_classes=2)),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, num_classes=2)),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, num_classes=2, pruning_threshold=0.02)),
    ]

    for model_name, model in models_config:
        print(f"\n>>> {model_name}")
        model.train()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(12):
            loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 128, device)
            if (epoch + 1) % 4 == 0:
                vm = evaluate_model(model, X_val, y_val, device)
                print(f"  Epoch {epoch+1}/12 - Loss: {loss:.4f}, Val Acc: {vm['accuracy']:.4f}")

        metrics = evaluate_model(model, X_test, y_test, device)
        model_size = compute_model_size_mb(model)
        num_params = count_parameters(model)
        inf_time = measure_inference_time(model, X_test, device)
        flops = estimate_flops(model, input_dim, device)

        sparsity = 0.0
        if hasattr(model, 'apply_pruning'):
            model.apply_pruning()
            sparsity = model.get_sparsity()

        print(f"  Test Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        print(f"  Size: {model_size:.3f} MB, Params: {num_params:,}, Inf: {inf_time:.3f} ms, FLOPs: {flops:.2f}M" +
              (f", Sparsity: {sparsity:.2%}" if sparsity else ""))

        results['models'][model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'auc': float(metrics['auc']),
            'model_size_mb': float(model_size),
            'num_parameters': int(num_params),
            'inference_time_ms': float(inf_time),
            'flops_millions': float(flops),
            'sparsity': float(sparsity),
        }

    return results


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    sample_config = {
        'WUSTL-IIoT': ('WUSTL-IIoT', 'wustl_iiot_2021.csv', None, 30000),
        'X-IIoTID': ('X-IIoTID', 'Binary-X-IIoTD.csv', 'class3', 30000),
        'Edge-IIoTset': ('Edge-IIoTset', 'edge_iiot_binary.csv', None, 30000),
    }

    all_results = {}
    for short_name, (folder, fname, label_col, nrows) in sample_config.items():
        path = os.path.join(data_dir, folder, fname)
        if os.path.exists(path):
            results = run_on_dataset(short_name, load_dataset, path, label_col, nrows)
            all_results[short_name] = results
        else:
            print(f"\nWARNING: {folder} not found at {path}")

    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")

    if all_results:
        first_dataset = list(all_results.keys())[0]
        model_names = list(all_results[first_dataset]['models'].keys())

        for metric_name in ['accuracy', 'f1_score', 'model_size_mb', 'num_parameters', 'inference_time_ms', 'flops_millions']:
            print(f"\n--- {metric_name.upper()} ---")
            header = f"{'Dataset':<20}"
            for m in model_names:
                short = m.split(' (')[0][:18]
                header += f" {short:<20}"
            print(header)
            print('-' * len(header))
            for dset, res in all_results.items():
                row = f"{dset:<20}"
                for m in model_names:
                    val = res['models'][m].get(metric_name, 'N/A')
                    if isinstance(val, float):
                        row += f" {val:<20.4f}"
                    else:
                        row += f" {str(val):<20}"
                print(row)

        print(f"\n{'='*70}")
        print("EFFICIENCY GAINS VS BASELINE (across datasets)")
        print(f"{'='*70}")
        for dataset_name, res in all_results.items():
            base = res['models'][model_names[0]]
            print(f"\n{dataset_name}:")
            for mname in model_names[1:]:
                m = res['models'][mname]
                sz = (1 - m['model_size_mb']/base['model_size_mb'])*100
                pr = (1 - m['num_parameters']/base['num_parameters'])*100
                fl = (1 - m['flops_millions']/base['flops_millions'])*100
                sp = base['inference_time_ms']/m['inference_time_ms']
                ad = (m['accuracy'] - base['accuracy'])*100
                print(f"  {mname.split(' (')[0]}: Size {sz:+.1f}%, Params {pr:+.1f}%, FLOPs {fl:+.1f}%, Speedup {sp:.2f}x, Acc {ad:+.2f}%")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'real_data_results.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")
