import sys, os, json, time, copy
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
from utils.metrics import evaluate_model, compute_model_size_mb, count_parameters, measure_inference_time, train_one_epoch

device = 'cpu'
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_file = os.path.join(base, 'results', 'quantization_results.json')


def get_quantized_model(model, input_dim):
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8
    )
    return quantized


def measure_model_size_mb(model):
    param_size = 0
    for p in model.parameters():
        if p.is_quantized:
            param_size += p.numel() * torch.iinfo(p.dtype).bits // 8
        else:
            param_size += p.numel() * (8 if p.element_size() == 1 else p.element_size())
    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.numel() * b.element_size()
    total_bytes = param_size + buffer_size
    return total_bytes / (1024 * 1024)


def load_generic(filepath, label_col=None, n_rows=20000):
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


def run_quantization_experiment(name, ModelClass, input_dim, X_train, y_train, X_test, y_test, extra_args=None, pretrained_state=None):
    print(f"\n  --- {name} ---")

    if extra_args:
        model = ModelClass(input_dim, 2, **extra_args)
    else:
        model = ModelClass(input_dim, 2)

    model.to(device)

    if pretrained_state is not None:
        model.load_state_dict(pretrained_state)
        model.eval()
        print(f"    Loaded pretrained model")
    else:
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for ep in range(10):
            loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 128, device)
            if (ep + 1) % 5 == 0:
                me = evaluate_model(model, X_test[:2000], y_test[:2000], device)
                print(f"    Epoch {ep+1}/10 - Loss: {loss:.4f}, Val Acc: {me['accuracy']:.4f}")

    fp32_metrics = evaluate_model(model, X_test, y_test, device)
    fp32_size = compute_model_size_mb(model)
    fp32_inf = measure_inference_time(model, X_test, device)
    fp32_params = sum(p.numel() for p in model.parameters())

    quantized = get_quantized_model(copy.deepcopy(model), input_dim)

    q_metrics = evaluate_model(quantized, X_test, y_test, device)
    q_size = measure_model_size_mb(quantized)
    q_inf = measure_inference_time(quantized, X_test, device)

    print(f"    FP32:  Acc={fp32_metrics['accuracy']:.4f} Size={fp32_size:.3f}MB Inf={fp32_inf:.2f}ms")
    print(f"    INT8:  Acc={q_metrics['accuracy']:.4f} Size={q_size:.3f}MB Inf={q_inf:.2f}ms")
    print(f"    Gains: Size {fp32_size/q_size:.1f}x  Speed {fp32_inf/q_inf:.1f}x")

    return {
        'fp32': {
            'accuracy': float(fp32_metrics['accuracy']),
            'f1_score': float(fp32_metrics['f1_score']),
            'model_size_mb': float(fp32_size),
            'num_parameters': int(fp32_params),
            'inference_time_ms': float(fp32_inf),
        },
        'int8': {
            'accuracy': float(q_metrics['accuracy']),
            'f1_score': float(q_metrics['f1_score']),
            'model_size_mb': float(q_size),
            'num_parameters': int(fp32_params),
            'inference_time_ms': float(q_inf),
        },
        'compression_ratio': float(fp32_size / q_size),
        'speedup': float(fp32_inf / q_inf),
        'accuracy_change': float(q_metrics['accuracy'] - fp32_metrics['accuracy']),
    }


if __name__ == '__main__':
    data_dir = os.path.join(base, 'data')

    datasets = [
        ('WUSTL-IIoT', os.path.join(data_dir, 'WUSTL-IIoT', 'wustl_iiot_2021.csv'), None, 20000),
        ('X-IIoTID', os.path.join(data_dir, 'X-IIoTID', 'Binary-X-IIoTD.csv'), 'class3', 20000),
    ]

    model_configs = [
        ('CNN-BiLSTM', CNNBiLSTM, {}),
        ('CNN-GRU (Lightweight)', LightweightCNNGRU, {}),
        ('Pruned CNN-GRU', PrunedLightweightCNNGRU, {'pruning_threshold': 0.02}),
    ]

    all_results = {}
    for dname, dpath, lcol, nrows in datasets:
        if not os.path.exists(dpath):
            print(f"SKIP {dname}: not found")
            continue

        print(f"\n{'='*60}")
        print(f"DATASET: {dname}")
        print(f"{'='*60}")
        X, y, input_dim = load_generic(dpath, lcol, nrows)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"  Samples: {len(X)}, Features: {input_dim}")

        # Phase 1: Train all models and save state dicts
        print(f"\n  --- Phase 1: Training models ---")
        pretrained_states = {}
        for mname, MClass, kwargs in model_configs:
            print(f"\n  --- Training {mname} ---")
            if kwargs:
                model = MClass(input_dim, 2, **kwargs)
            else:
                model = MClass(input_dim, 2)
            model.to(device)
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for ep in range(10):
                loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 128, device)
                if (ep + 1) % 5 == 0:
                    me = evaluate_model(model, X_test[:2000], y_test[:2000], device)
                    print(f"    Epoch {ep+1}/10 - Loss: {loss:.4f}, Val Acc: {me['accuracy']:.4f}")
            pretrained_states[mname] = copy.deepcopy(model.state_dict())

        # Phase 2: Run quantization using pretrained models
        print(f"\n  --- Phase 2: Quantization ---")
        dataset_results = {}
        for mname, MClass, kwargs in model_configs:
            res = run_quantization_experiment(
                mname, MClass, input_dim,
                X_train, y_train, X_test, y_test,
                kwargs,
                pretrained_state=pretrained_states[mname]
            )
            dataset_results[mname] = res

        all_results[dname] = dataset_results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*60}")
    for dn, res in all_results.items():
        print(f"\n  {dn}:")
        for mn, m in res.items():
            print(f"    {mn:<25} FP32: {m['fp32']['model_size_mb']:.3f}MB/{m['fp32']['inference_time_ms']:.2f}ms "
                  f"INT8: {m['int8']['model_size_mb']:.3f}MB/{m['int8']['inference_time_ms']:.2f}ms "
                  f"({m['compression_ratio']:.1f}x size, {m['speedup']:.1f}x speed, "
                  f"acc: {m['accuracy_change']:+.4f})")

    print(f"\nResults saved to {results_file}")
