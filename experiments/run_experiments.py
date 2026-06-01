import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import json
import copy
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.synthetic_data import generate_synthetic_iiot_data, generate_synthetic_non_iid_clients
from utils.data_loader import split_data, create_federated_clients
from utils.metrics import (
    evaluate_model, compute_model_size_mb, count_parameters,
    measure_inference_time, estimate_flops, train_one_epoch, federated_averaging
)


def train_centralized(model, X_train, y_train, X_val, y_val,
                      epochs=30, batch_size=64, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, batch_size, device)
        val_metrics = evaluate_model(model, X_val, y_val, device)
        val_acc = val_metrics['accuracy']
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


def federated_train(global_model, client_data, X_test, y_test,
                    num_rounds=20, local_epochs=3, batch_size=64,
                    lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    client_models = [copy.deepcopy(global_model) for _ in range(len(client_data))]
    history = {'round_acc': [], 'communication_cost_mb': []}
    total_params_bytes = sum(p.numel() * p.element_size() for p in global_model.parameters())

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
        comm_cost = (total_params_bytes * len(client_data) * (round_num + 1)) / (1024 * 1024)
        history['round_acc'].append(test_metrics['accuracy'])
        history['communication_cost_mb'].append(comm_cost)
        if (round_num + 1) % 5 == 0:
            print(f"  Round {round_num+1}/{num_rounds} - Acc: {test_metrics['accuracy']:.4f}, Comm: {comm_cost:.2f} MB")

    return global_model, history


def run_experiment(device='cpu'):
    print("=" * 70)
    print("IIoT INTRUSION DETECTION - LIGHTWEIGHT MODEL STUDY")
    print("=" * 70)

    print("\n[1] Generating synthetic IIoT dataset...")
    X, y = generate_synthetic_iiot_data(n_samples=50000, n_features=65, random_state=42)
    print(f"    Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"    Normal: {np.sum(y==0)}, Attack: {np.sum(y==1)}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    input_dim = X.shape[1]
    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    all_results = {}

    print("\n" + "=" * 70)
    print("PART A: CENTRALIZED TRAINING")
    print("=" * 70)

    models_config = [
        ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, num_classes=2)),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, num_classes=2)),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, num_classes=2, pruning_threshold=0.02)),
    ]

    centralized_results = {}
    for model_name, model in models_config:
        print(f"\n>>> Training: {model_name}")
        trained_model, history = train_centralized(
            model, X_train, y_train, X_val, y_val,
            epochs=25, batch_size=64, device=device
        )

        test_metrics = evaluate_model(trained_model, X_test, y_test, device)
        model_size = compute_model_size_mb(trained_model)
        num_params = count_parameters(trained_model)
        inf_time = measure_inference_time(trained_model, X_test, device)
        flops = estimate_flops(trained_model, input_dim, device)

        sparsity = 0.0
        if hasattr(trained_model, 'apply_pruning'):
            trained_model.apply_pruning()
            sparsity = trained_model.get_sparsity()

        print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"    Precision: {test_metrics['precision']:.4f}")
        print(f"    Recall:    {test_metrics['recall']:.4f}")
        print(f"    F1-Score:  {test_metrics['f1_score']:.4f}")
        print(f"    AUC:       {test_metrics['auc']:.4f}")
        print(f"    Model Size: {model_size:.3f} MB")
        print(f"    Parameters: {num_params:,}")
        print(f"    Inference:  {inf_time:.3f} ms/sample")
        print(f"    FLOPs:      {flops:.2f}M")
        if sparsity > 0:
            print(f"    Sparsity:   {sparsity:.2%}")

        centralized_results[model_name] = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'auc': test_metrics['auc'],
            'model_size_mb': model_size,
            'num_parameters': num_params,
            'inference_time_ms': inf_time,
            'flops_millions': flops,
            'sparsity': sparsity,
        }

    all_results['centralized'] = centralized_results

    print("\n" + "=" * 70)
    print("PART B: FEDERATED LEARNING (IID)")
    print("=" * 70)

    X_train_all, X_test_fl, y_train_all, y_test_fl = \
        train_test_split_no_val(X, y)

    client_data_iid = create_federated_clients(X_train_all, y_train_all,
                                                num_clients=5, non_iid=False)

    fl_iid_results = {}
    for model_name, model in models_config:
        print(f"\n>>> FL (IID) Training: {model_name}")
        global_model, history = federated_train(
            model, client_data_iid, X_test_fl, y_test_fl,
            num_rounds=20, local_epochs=3, device=device
        )
        test_metrics = evaluate_model(global_model, X_test_fl, y_test_fl, device)

        print(f"    Final Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Final F1:       {test_metrics['f1_score']:.4f}")

        fl_iid_results[model_name] = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'auc': test_metrics['auc'],
            'round_accuracies': history['round_acc'],
            'communication_cost_mb': history['communication_cost_mb'],
        }

    all_results['fl_iid'] = fl_iid_results

    print("\n" + "=" * 70)
    print("PART C: FEDERATED LEARNING (Non-IID)")
    print("=" * 70)

    client_data_non_iid = create_federated_clients(X_train_all, y_train_all,
                                                    num_clients=5, non_iid=True)

    fl_non_iid_results = {}
    for model_name, model in models_config:
        print(f"\n>>> FL (Non-IID) Training: {model_name}")
        global_model, history = federated_train(
            model, client_data_non_iid, X_test_fl, y_test_fl,
            num_rounds=20, local_epochs=3, device=device
        )
        test_metrics = evaluate_model(global_model, X_test_fl, y_test_fl, device)

        print(f"    Final Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Final F1:       {test_metrics['f1_score']:.4f}")

        fl_non_iid_results[model_name] = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score'],
            'auc': test_metrics['auc'],
            'round_accuracies': history['round_acc'],
            'communication_cost_mb': history['communication_cost_mb'],
        }

    all_results['fl_non_iid'] = fl_non_iid_results

    save_results(all_results)
    generate_tables(all_results)

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 70)

    return all_results


def train_test_split_no_val(X, y, test_size=0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def save_results(results):
    results_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'results', 'experiment_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


def generate_tables(results):
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Centralized Training")
    print("=" * 70)
    print(f"{'Model':<30} {'Acc':<8} {'F1':<8} {'Size(MB)':<10} {'Params':<12} {'Inf(ms)':<10} {'FLOPs(M)':<10} {'Sparsity':<10}")
    print("-" * 98)
    for model_name, metrics in results['centralized'].items():
        print(f"{model_name:<30} {metrics['accuracy']:<8.4f} {metrics['f1_score']:<8.4f} "
              f"{metrics['model_size_mb']:<10.3f} {metrics['num_parameters']:<12,} "
              f"{metrics['inference_time_ms']:<10.3f} {metrics['flops_millions']:<10.2f} "
              f"{metrics['sparsity']:<10.2%}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Federated Learning")
    print("=" * 70)
    print(f"{'Model':<30} {'IID Acc':<10} {'IID F1':<10} {'Non-IID Acc':<12} {'Non-IID F1':<10}")
    print("-" * 72)
    for model_name in results['centralized'].keys():
        iid = results['fl_iid'].get(model_name, {})
        non_iid = results['fl_non_iid'].get(model_name, {})
        print(f"{model_name:<30} {iid.get('accuracy', 0):<10.4f} {iid.get('f1_score', 0):<10.4f} "
              f"{non_iid.get('accuracy', 0):<12.4f} {non_iid.get('f1_score', 0):<10.4f}")

    print("\n" + "=" * 70)
    print("EFFICIENCY COMPARISON (vs Baseline)")
    print("=" * 70)
    baseline = results['centralized']['CNN-BiLSTM (Baseline)']
    for model_name, metrics in results['centralized'].items():
        if model_name == baseline:
            continue
        size_reduction = (1 - metrics['model_size_mb'] / baseline['model_size_mb']) * 100
        speedup = baseline['inference_time_ms'] / metrics['inference_time_ms']
        param_reduction = (1 - metrics['num_parameters'] / baseline['num_parameters']) * 100
        flops_reduction = (1 - metrics['flops_millions'] / baseline['flops_millions']) * 100
        acc_diff = metrics['accuracy'] - baseline['accuracy']
        print(f"\n{model_name}:")
        print(f"  Model Size:  {size_reduction:+.1f}%")
        print(f"  Parameters:  {param_reduction:+.1f}%")
        print(f"  FLOPs:       {flops_reduction:+.1f}%")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Accuracy:    {acc_diff:+.4f} vs baseline")

    try:
        plot_results(results, output_dir)
    except Exception as e:
        print(f"Note: Could not generate plots: {e}")


def plot_results(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = list(results['centralized'].keys())
    accs = [results['centralized'][m]['accuracy'] for m in models]
    f1s = [results['centralized'][m]['f1_score'] for m in models]
    sizes = [results['centralized'][m]['model_size_mb'] for m in models]
    times = [results['centralized'][m]['inference_time_ms'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax1 = axes[0, 0]
    ax1.bar(x - width/2, accs, width, label='Accuracy', color='steelblue')
    ax1.bar(x + width/2, f1s, width, label='F1-Score', color='lightcoral')
    ax1.set_ylabel('Score')
    ax1.set_title('Accuracy & F1-Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.split(' (')[0] for m in models], rotation=15)
    ax1.legend()
    ax1.set_ylim(0.9, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[0, 1]
    ax2.bar(x, sizes, color=['steelblue', 'seagreen', 'orange'])
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split(' (')[0] for m in models], rotation=15)
    ax2.grid(axis='y', alpha=0.3)

    ax3 = axes[1, 0]
    ax3.bar(x, times, color=['steelblue', 'seagreen', 'orange'])
    ax3.set_ylabel('Inference Time (ms)')
    ax3.set_title('Inference Speed Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.split(' (')[0] for m in models], rotation=15)
    ax3.grid(axis='y', alpha=0.3)

    ax4 = axes[1, 1]
    for model_name in models:
        if model_name in results['fl_iid']:
            accs_iid = results['fl_iid'][model_name]['round_accuracies']
            ax4.plot(accs_iid, label=f"{model_name.split(' (')[0]}", marker='o', markersize=3, linewidth=1.5)
    ax4.set_xlabel('Federated Round')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Federated Learning Convergence')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'figures', 'comparison_plots.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    run_experiment(device='cpu')
