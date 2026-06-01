import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(model, X_test, y_test, device='cpu', average='binary'):
    model.eval()
    model.to(device)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_tensor = torch.LongTensor(y_test).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        if outputs.shape[1] == 2:
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        else:
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

    y_pred = predicted.cpu().numpy()
    y_true = y_test

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0, average=average),
        'recall': recall_score(y_true, y_pred, zero_division=0, average=average),
        'f1_score': f1_score(y_true, y_pred, zero_division=0, average=average),
    }
    if average == 'binary':
        try:
            metrics['auc'] = roc_auc_score(y_true, probs)
        except Exception:
            metrics['auc'] = 0.0

    return metrics


def compute_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_size + buffer_size
    return total_bytes / (1024 * 1024)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(model, X_test, device='cpu', num_runs=100):
    model.eval()
    model.to(device)
    X_tensor = torch.FloatTensor(X_test[:100]).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(X_tensor)

    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(X_tensor)
    end = time.time()

    avg_time_ms = ((end - start) / num_runs) * 1000
    return avg_time_ms


def estimate_flops(model, input_dim, device='cpu'):
    try:
        from thop import profile
        model.eval()
        dummy_input = torch.randn(1, input_dim).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e6
    except Exception:
        return 0.0


def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_models[i].state_dict()[key].float() for i in range(len(client_models))],
            dim=0
        ).mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model


def train_one_epoch(model, X_train, y_train, optimizer, criterion, batch_size=64, device='cpu'):
    model.train()
    model.to(device)
    n = len(X_train)
    indices = np.random.permutation(n)
    total_loss = 0
    num_batches = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        X_batch = torch.FloatTensor(X_train[batch_idx]).to(device)
        y_batch = torch.LongTensor(y_train[batch_idx]).to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0
