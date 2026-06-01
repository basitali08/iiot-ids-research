import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
import copy
import time

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.data_loader import (
    load_and_preprocess_edge_iiotset,
    load_and_preprocess_xiiotid,
    load_and_preprocess_wustl,
    create_federated_clients
)
from utils.metrics import (
    evaluate_model,
    compute_model_size_mb,
    count_parameters,
    measure_inference_time,
    estimate_flops,
    train_one_epoch,
    federated_averaging
)


def federated_train(global_model, client_data, X_test, y_test,
                    num_rounds=20, local_epochs=3, batch_size=64,
                    lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    client_models = []
    for _ in range(len(client_data)):
        client_models.append(copy.deepcopy(global_model))

    history = {'round_acc': [], 'communication_cost_mb': []}

    total_params = sum(p.numel() * p.element_size() for p in global_model.parameters())

    for round_num in range(num_rounds):
        for client_idx, (X_c, y_c) in enumerate(client_data):
            model = client_models[client_idx]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for _ in range(local_epochs):
                train_one_epoch(model, X_c, y_c, optimizer, criterion, batch_size, device)

        global_model = federated_averaging(global_model, client_models)

        for cm in client_models:
            cm.load_state_dict(global_model.state_dict())

        test_metrics = evaluate_model(global_model, X_test, y_test, device)

        comm_cost = (total_params * len(client_data) * (round_num + 1)) / (1024 * 1024)
        history['round_acc'].append(test_metrics['accuracy'])
        history['communication_cost_mb'].append(comm_cost)

        if (round_num + 1) % 5 == 0:
            print(f"  Round {round_num+1}/{num_rounds} - Test Acc: {test_metrics['accuracy']:.4f}, "
                  f"Comm: {comm_cost:.2f} MB")

    return global_model, history


def run_federated_experiment(dataset_name, load_fn, filepath, num_clients=5,
                             non_iid=False, sample_frac=0.3, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Federated - {dataset_name} ({'non-IID' if non_iid else 'IID'})")
    print(f"{'='*60}")

    print("Loading data...")
    X, y, _ = load_fn(filepath, sample_frac=sample_frac)
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    client_data = create_federated_clients(X_train, y_train, num_clients=num_clients,
                                           non_iid=non_iid, random_state=42)

    input_dim = X.shape[1]
    results = {'dataset': dataset_name, 'num_clients': num_clients,
               'non_iid': non_iid, 'models': {}}

    models_config = [
        ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, num_classes=2)),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, num_classes=2)),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, num_classes=2)),
    ]

    for model_name, model in models_config:
        print(f"\n--- FL Training: {model_name} ---")
        global_model, history = federated_train(
            model, client_data, X_test, y_test,
            num_rounds=20, local_epochs=3, device=device
        )

        test_metrics = evaluate_model(global_model, X_test, y_test, device)
        model_size = compute_model_size_mb(global_model)
        num_params = count_parameters(global_model)
        inf_time = measure_inference_time(global_model, X_test, device)
        flops = estimate_flops(global_model, input_dim, device)

        sparsity = 0.0
        if hasattr(global_model, 'get_sparsity'):
            sparsity = global_model.get_sparsity()

        print(f"  Final - Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_score']:.4f}")
        print(f"  Size: {model_size:.3f} MB | Params: {num_params:,} | Inf: {inf_time:.3f} ms")

        results['models'][model_name] = {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'model_size_mb': float(model_size),
            'num_parameters': int(num_params),
            'inference_time_ms': float(inf_time),
            'flops_millions': float(flops),
            'sparsity': float(sparsity),
            'round_accuracies': [float(a) for a in history['round_acc']],
            'communication_cost_mb': [float(c) for c in history['communication_cost_mb']],
            'final_communication_cost_mb': float(history['communication_cost_mb'][-1]),
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

    all_results = {}
    for name, load_fn, path in datasets_config:
        for non_iid in [False, True]:
            results = run_federated_experiment(name, load_fn, path,
                                               num_clients=5, non_iid=non_iid,
                                               sample_frac=0.3, device=device)
            key = f"{name}_{'non_iid' if non_iid else 'iid'}"
            all_results[key] = results

    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'federated_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")
