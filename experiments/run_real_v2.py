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


def load_and_train(name, filepath, label_col, n_rows, bsize=128, epochs=10):
    print(f"\n{'='*60}")
    print(f"LOADING: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath, low_memory=False, nrows=n_rows)
    print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} cols")

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

    print(f"  Features: {X.shape[1]}, Attack ratio: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_dim = X.shape[1]
    configs = [
        ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, 2)),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, 2)),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, 2, pruning_threshold=0.02)),
    ]

    results = {}
    for mname, model in configs:
        print(f"\n>>> {mname}")
        model.train()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for ep in range(epochs):
            loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, bsize, device)
            if (ep + 1) % 5 == 0:
                vm = evaluate_model(model, X_val, y_val, device)
                print(f"  Epoch {ep+1}/{epochs} - Loss: {loss:.4f}, Val Acc: {vm['accuracy']:.4f}")

        me = evaluate_model(model, X_test, y_test, device)
        sz = compute_model_size_mb(model)
        pa = count_parameters(model)
        ti = measure_inference_time(model, X_test, device)
        fl = estimate_flops(model, input_dim, device)

        sp = 0.0
        if hasattr(model, 'apply_pruning'):
            model.apply_pruning()
            sp = model.get_sparsity()

        print(f"  Test -> Acc: {me['accuracy']:.4f}, F1: {me['f1_score']:.4f}")
        print(f"  Size: {sz:.3f} MB | Params: {pa:,} | Inf: {ti:.3f} ms | FLOPs: {fl:.2f}M" +
              (f" | Sparsity: {sp:.2%}" if sp else ""))

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

    return {name: { 'input_dim': input_dim, 'models': results }}


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'real_data_results.json')

    jobs = [
        ('WUSTL-IIoT', os.path.join(data_dir, 'WUSTL-IIoT', 'wustl_iiot_2021.csv'), None, 20000),
        ('X-IIoTID', os.path.join(data_dir, 'X-IIoTID', 'Binary-X-IIoTD.csv'), 'class3', 20000),
        ('Edge-IIoTset', os.path.join(data_dir, 'Edge-IIoTset', 'edge_iiot_binary.csv'), None, 20000),
    ]

    all_results = {}
    for dname, dpath, lcol, nrows in jobs:
        if os.path.exists(dpath):
            res = load_and_train(dname, dpath, lcol, nrows, epochs=10)
            all_results.update(res)
            with open(outpath, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nSaved partial results to {outpath}")
        else:
            print(f"\nSKIP {dname}: not found at {dpath}")

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for dn, res in all_results.items():
        print(f"\n--- {dn} ---")
        b = list(res['models'].values())[0]
        for mn, m in res['models'].items():
            short = mn.split(' (')[0]
            sz_red = (1 - m['model_size_mb']/b['model_size_mb'])*100
            sp_up = b['inference_time_ms']/m['inference_time_ms']
            print(f"  {short:<20} Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} "
                  f"Size={m['model_size_mb']:.3f}MB Params={m['num_parameters']:,} "
                  f"Inf={m['inference_time_ms']:.2f}ms FLOPs={m['flops_millions']:.2f}M "
                  + (f"Sparsity={m['sparsity']:.1%}" if m['sparsity'] > 0 else "")
                  + (f" ({sz_red:+.1f}%, {sp_up:.2f}x)" if 'Baseline' not in mn else ""))

    print(f"\nAll results saved to {outpath}")
