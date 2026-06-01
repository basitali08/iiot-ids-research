import sys, os, json, copy, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.synthetic_data import generate_synthetic_iiot_data
from utils.metrics import (
    evaluate_model, compute_model_size_mb, count_parameters,
    measure_inference_time, estimate_flops, train_one_epoch
)

device = 'cpu'
print('Generating synthetic IIoT data...')
X, y = generate_synthetic_iiot_data(n_samples=10000, n_features=65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

models = {
    'CNN-BiLSTM (Baseline)': CNNBiLSTM(65, 2),
    'CNN-GRU (Lightweight)': LightweightCNNGRU(65, 2),
    'Pruned CNN-GRU': PrunedLightweightCNNGRU(65, 2, pruning_threshold=0.02),
}

results = {}
for name, model in models.items():
    print(f'\n{"="*60}')
    print(f'Training: {name}')
    print(f'{"="*60}')
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(15):
        loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 64, device)
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_model(model, X_val, y_val, device)
            print(f'  Epoch {epoch+1}/15 - Loss: {loss:.4f}, Val Acc: {metrics["accuracy"]:.4f}')

    metrics = evaluate_model(model, X_test, y_test, device)
    size = compute_model_size_mb(model)
    params = count_parameters(model)
    inf_time = measure_inference_time(model, X_test, device)
    flops = estimate_flops(model, 65, device)

    sparsity = 0.0
    if hasattr(model, 'apply_pruning'):
        model.apply_pruning()
        sparsity = model.get_sparsity()

    print(f'\n  Results:')
    print(f'    Accuracy:  {metrics["accuracy"]:.4f}')
    print(f'    Precision: {metrics["precision"]:.4f}')
    print(f'    Recall:    {metrics["recall"]:.4f}')
    print(f'    F1-Score:  {metrics["f1_score"]:.4f}')
    print(f'    AUC:       {metrics["auc"]:.4f}')
    print(f'    Model Size: {size:.3f} MB')
    print(f'    Parameters: {params:,}')
    print(f'    Inf Time:  {inf_time:.3f} ms/sample')
    print(f'    FLOPs:     {flops:.2f}M')
    if sparsity > 0:
        print(f'    Sparsity:  {sparsity:.2%}')

    results[name] = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'auc': float(metrics['auc']),
        'model_size_mb': float(size),
        'num_parameters': int(params),
        'inference_time_ms': float(inf_time),
        'flops_millions': float(flops),
        'sparsity': float(sparsity),
    }

print(f'\n{"="*70}')
print('CENTRALIZED RESULTS SUMMARY')
print(f'{"="*70}')
header = f'{"Model":<30} {"Acc":<8} {"F1":<8} {"Size(MB)":<10} {"Params":<12} {"Inf(ms)":<10} {"FLOPs(M)":<10}'
print(header)
print('-' * len(header))
for name, r in results.items():
    print(f'{name:<30} {r["accuracy"]:<8.4f} {r["f1_score"]:<8.4f} '
          f'{r["model_size_mb"]:<10.3f} {r["num_parameters"]:<12,} '
          f'{r["inference_time_ms"]:<10.3f} {r["flops_millions"]:<10.2f}')

print(f'\n{"="*70}')
print('EFFICIENCY VS BASELINE')
print(f'{"="*70}')
baseline = results['CNN-BiLSTM (Baseline)']
for name, r in results.items():
    if 'Baseline' in name:
        continue
    size_red = (1 - r['model_size_mb'] / baseline['model_size_mb']) * 100
    param_red = (1 - r['num_parameters'] / baseline['num_parameters']) * 100
    flops_red = (1 - r['flops_millions'] / baseline['flops_millions']) * 100
    speedup = baseline['inference_time_ms'] / r['inference_time_ms']
    acc_diff = (r['accuracy'] - baseline['accuracy']) * 100
    print(f'\n{name}:')
    print(f'  Size:       {size_red:+.1f}%')
    print(f'  Params:     {param_red:+.1f}%')
    print(f'  FLOPs:      {flops_red:+.1f}%')
    print(f'  Speedup:    {speedup:.2f}x')
    print(f'  Accuracy:   {acc_diff:+.2f}% vs baseline')

results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'centralized_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {results_dir}/centralized_results.json')
