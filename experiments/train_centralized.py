import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
import time

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.data_loader import (
    load_and_preprocess_edge_iiotset,
    load_and_preprocess_xiiotid,
    load_and_preprocess_wustl,
    split_data
)
from utils.metrics import (
    evaluate_model,
    compute_model_size_mb,
    count_parameters,
    measure_inference_time,
    estimate_flops,
    train_one_epoch
)


def train_centralized(model, X_train, y_train, X_val, y_val,
                      epochs=30, batch_size=64, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, batch_size, device)
        val_metrics = evaluate_model(model, X_val, y_val, device)
        val_acc = val_metrics['accuracy']
        scheduler.step(val_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def run_experiment(dataset_name, load_fn, filepath, sample_frac=1.0, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    print("Loading and preprocessing data...")
    X, y, scaler = load_fn(filepath, sample_frac=sample_frac)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Class distribution: normal={np.sum(y==0)}, attack={np.sum(y==1)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    input_dim = X.shape[1]
    results = {'dataset': dataset_name, 'input_dim': input_dim, 'models': {}}

    models_config = [
        ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, num_classes=2)),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, num_classes=2)),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, num_classes=2)),
    ]

    for model_name, model in models_config:
        print(f"\n--- Training: {model_name} ---")

        trained_model, history = train_centralized(
            model, X_train, y_train, X_val, y_val,
            epochs=30, batch_size=64, device=device
        )

        test_metrics = evaluate_model(trained_model, X_test, y_test, device)
        model_size = compute_model_size_mb(trained_model)
        num_params = count_parameters(trained_model)
        inf_time = measure_inference_time(trained_model, X_test, device)
        flops = estimate_flops(trained_model, input_dim, device)

        sparsity = 0.0
        if hasattr(trained_model, 'get_sparsity'):
            sparsity = trained_model.get_sparsity()

        print(f"  Test Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_score']:.4f}")
        print(f"  Size: {model_size:.3f} MB | Params: {num_params:,} | FLOPs: {flops:.2f}M")
        print(f"  Inference: {inf_time:.3f} ms/sample | Sparsity: {sparsity:.2%}")

        results['models'][model_name] = {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'model_size_mb': float(model_size),
            'num_parameters': int(num_params),
            'inference_time_ms': float(inf_time),
            'flops_millions': float(flops),
            'sparsity': float(sparsity),
        }

    return results


if __name__ == '__main__':
    device = 'cpu'
    print(f"Using device: {device}")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    datasets_config = []

    edge_path = os.path.join(data_dir, 'Edge-IIoTset', 'DNN-EdgeIIoT-dataset.csv')
    if os.path.exists(edge_path):
        datasets_config.append(('Edge-IIoTset', load_and_preprocess_edge_iiotset, edge_path))

    xiiotid_path = os.path.join(data_dir, 'X-IIoTID', 'X-IIoTID.csv')
    if os.path.exists(xiiotid_path):
        datasets_config.append(('X-IIoTID', load_and_preprocess_xiiotid, xiiotid_path))

    wustl_path = os.path.join(data_dir, 'WUSTL-IIoT', 'wustl_iiot_2021.csv')
    if os.path.exists(wustl_path):
        datasets_config.append(('WUSTL-IIoT', load_and_preprocess_wustl, wustl_path))

    if not datasets_config:
        print("No datasets found. Checking for alternative file names...")
        for f in os.listdir(data_dir):
            print(f"  Found: {f}")

    all_results = {}
    for name, load_fn, path in datasets_config:
        results = run_experiment(name, load_fn, path, sample_frac=0.3, device=device)
        all_results[name] = results

    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'centralized_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")
