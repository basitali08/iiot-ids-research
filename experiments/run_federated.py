import sys, os, json, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.synthetic_data import generate_synthetic_iiot_data
from utils.data_loader import create_federated_clients
from utils.metrics import (
    evaluate_model, compute_model_size_mb, count_parameters,
    measure_inference_time, estimate_flops, train_one_epoch,
    federated_averaging
)

device = 'cpu'
print('Generating synthetic IIoT data for FL experiments...')
X, y = generate_synthetic_iiot_data(n_samples=20000, n_features=65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

for non_iid in [False, True]:
    mode = 'Non-IID' if non_iid else 'IID'
    print(f'\n{"="*70}')
    print(f'FEDERATED LEARNING ({mode})')
    print(f'{"="*70}')

    client_data = create_federated_clients(X_train, y_train, num_clients=5, non_iid=non_iid)
    for i, (Xc, yc) in enumerate(client_data):
        attack_ratio = yc.mean() * 100
        print(f'  Client {i+1}: {len(Xc)} samples, {attack_ratio:.1f}% attacks')

    models = {
        'CNN-BiLSTM (Baseline)': CNNBiLSTM(65, 2),
        'CNN-GRU (Lightweight)': LightweightCNNGRU(65, 2),
        'Pruned CNN-GRU': PrunedLightweightCNNGRU(65, 2, pruning_threshold=0.02),
    }

    results = {}
    for name, model in models.items():
        print(f'\n--- FL {mode}: {name} ---')
        global_model = copy.deepcopy(model)
        client_models = [copy.deepcopy(global_model) for _ in range(5)]
        criterion = nn.CrossEntropyLoss()

        round_accs = []
        for round_num in range(15):
            for client_idx, (X_c, y_c) in enumerate(client_data):
                cm = client_models[client_idx]
                opt = torch.optim.Adam(cm.parameters(), lr=0.001)
                for _ in range(3):
                    train_one_epoch(cm, X_c, y_c, opt, criterion, 64, device)

            global_model = federated_averaging(global_model, client_models)
            for cm in client_models:
                cm.load_state_dict(global_model.state_dict())

            metrics = evaluate_model(global_model, X_test, y_test, device)
            round_accs.append(float(metrics['accuracy']))

            if (round_num + 1) % 5 == 0:
                print(f'  Round {round_num+1}/15 - Acc: {metrics["accuracy"]:.4f}, F1: {metrics["f1_score"]:.4f}')

        total_comm_mb = round_accs[-1]
        print(f'  Final - Acc: {metrics["accuracy"]:.4f}, F1: {metrics["f1_score"]:.4f}')
        results[name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'auc': float(metrics['auc']),
            'round_accuracies': round_accs,
        }

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    with open(os.path.join(results_dir, f'fl_{mode.lower()}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nFL {mode} SUMMARY:')
    print(f'{"Model":<30} {"Accuracy":<10} {"F1-Score":<10}')
    print('-' * 50)
    for name, r in results.items():
        print(f'{name:<30} {r["accuracy"]:<10.4f} {r["f1_score"]:<10.4f}')

print(f'\n{"="*70}')
print('FINAL COMPARISON: CENTRALIZED vs FEDERATED')
print(f'{"="*70}')
with open(os.path.join(results_dir, 'centralized_results.json')) as f:
    cen = json.load(f)
with open(os.path.join(results_dir, 'fl_iid_results.json')) as f:
    fl_iid = json.load(f)
with open(os.path.join(results_dir, 'fl_non_iid_results.json')) as f:
    fl_nid = json.load(f)

print(f'{"Model":<25} {"Centralized":<15} {"FL IID":<15} {"FL Non-IID":<15}')
print(f'{"":<25} {"Acc    F1":<15} {"Acc    F1":<15} {"Acc    F1":<15}')
print('-' * 70)
for name in cen.keys():
    short = name.split(' (')[0]
    c = cen[name]
    i = fl_iid.get(name, {})
    n = fl_nid.get(name, {})
    print(f'{short:<25} {c["accuracy"]:.4f} {c["f1_score"]:.4f}  {i.get("accuracy",0):.4f} {i.get("f1_score",0):.4f}  {n.get("accuracy",0):.4f} {n.get("f1_score",0):.4f}')

print(f'\nAll results saved to {results_dir}/')
