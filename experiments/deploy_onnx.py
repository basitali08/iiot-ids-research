"""
Deploy models to ONNX for edge inference.
Trains models, exports to ONNX, validates correctness, benchmarks latency.
"""

import sys, os, json, time, copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.metrics import train_one_epoch, evaluate_model, compute_model_size_mb

DEVICE = 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE, 'results', 'deployed')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_xiitod(path, n_rows=20000):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(path, low_memory=False, nrows=n_rows)
    label_col = 'class3' if 'class3' in df.columns else df.columns[-1]
    df['label'] = df[label_col].apply(
        lambda x: 0 if str(x).strip().lower() in ['0', 'normal', 'benign', ''] else 1
    )
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['label', label_col]]
    for c in df.select_dtypes(include=['object', 'str']).columns.tolist():
        if c != label_col:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    feature_cols = [c for c in df.columns if c not in ['label', label_col]]
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    X = StandardScaler().fit_transform(X)
    return X, y, X.shape[1]


def train_model(ModelClass, input_dim, num_classes, X_train, y_train, X_test, y_test, extra_kwargs=None):
    kwargs = extra_kwargs or {}
    model = ModelClass(input_dim, num_classes, **kwargs)
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for ep in range(10):
        loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 128, DEVICE)
        if (ep + 1) % 5 == 0:
            me = evaluate_model(model, X_test[:2000], y_test[:2000], DEVICE)
            print(f"    Epoch {ep+1}/10 - Loss: {loss:.4f}, Val Acc: {me['accuracy']:.4f}")
    model.eval()
    return model


def export_to_onnx(model, input_dim, filepath, num_classes=2):
    model.eval()
    dummy = torch.randn(1, input_dim)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            filepath,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    return os.path.getsize(filepath)


def test_onnx_inference(filepath, X_test, num_runs=100):
    import onnxruntime as ort
    session = ort.InferenceSession(filepath)
    input_name = session.get_inputs()[0].name

    # Warmup
    _ = session.run(None, {input_name: X_test[:1].astype(np.float32)})

    # Benchmark
    start = time.perf_counter()
    for i in range(min(num_runs, len(X_test))):
        _ = session.run(None, {input_name: X_test[i:i+1].astype(np.float32)})
    avg_ms = (time.perf_counter() - start) / min(num_runs, len(X_test)) * 1000

    # Accuracy check
    all_preds = []
    batch_size = 256
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].astype(np.float32)
        out = session.run(None, {input_name: batch})[0]
        all_preds.append(np.argmax(out, axis=1))
    preds = np.concatenate(all_preds)

    return avg_ms, preds


def benchmark_edge(filepath, input_dim, num_runs=500):
    """Simulate edge deployment conditions."""
    import onnxruntime as ort
    session = ort.InferenceSession(filepath)
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(1, input_dim).astype(np.float32)
    for _ in range(10):
        session.run(None, {input_name: dummy})

    times = []
    dummy = np.random.randn(1, input_dim).astype(np.float32)
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'samples_per_sec': float(1000 / np.mean(times)),
        'num_runs': num_runs,
    }


if __name__ == '__main__':
    data_path = os.path.join(BASE, 'data', 'X-IIoTID', 'Binary-X-IIoTD.csv')
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("ONNX DEPLOYMENT PIPELINE")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    X, y, input_dim = load_xiitod(data_path, 20000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Input dim: {input_dim}, Train: {len(X_train)}, Test: {len(X_test)}")

    model_configs = [
        ('CNN-GRU_Light', LightweightCNNGRU, {}, 2),
        ('CNN-GRU_Pruned', PrunedLightweightCNNGRU, {'pruning_threshold': 0.02}, 2),
    ]

    all_results = {}
    for name, MClass, kwargs, num_cls in model_configs:
        print(f"\n[2/4] Training {name}...")
        model = train_model(MClass, input_dim, num_cls, X_train, y_train, X_test, y_test, kwargs)

        if 'Pruned' in name:
            model.apply_pruning()
            sparsity = model.get_sparsity()
            print(f"  Pruning applied: sparsity={sparsity:.2%}")

        pt_path = os.path.join(OUTPUT_DIR, f'{name}.pt')
        torch.save(model.state_dict(), pt_path)
        pt_size = os.path.getsize(pt_path) / (1024 * 1024)

        onnx_path = os.path.join(OUTPUT_DIR, f'{name}.onnx')
        print(f"\n[3/4] Exporting {name} to ONNX...")
        onnx_size_bytes = export_to_onnx(model, input_dim, onnx_path, num_cls)
        onnx_size_mb = onnx_size_bytes / (1024 * 1024)

        print(f"\n[4/4] Benchmarking {name}...")
        pt_metrics = evaluate_model(model, X_test, y_test, DEVICE)
        pt_inf = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                model(torch.from_numpy(X_test[:1].astype(np.float32)))
        pt_inf = (time.perf_counter() - pt_inf) / 100 * 1000

        onnx_ms, onnx_preds = test_onnx_inference(onnx_path, X_test, 100)
        onnx_acc = np.mean(onnx_preds == y_test)

        edge = benchmark_edge(onnx_path, input_dim, 500)

        result = {
            'pytorch': {
                'accuracy': float(pt_metrics['accuracy']),
                'f1_score': float(pt_metrics['f1_score']),
                'model_size_mb': round(pt_size, 3),
                'inference_time_ms': round(pt_inf, 2),
            },
            'onnx': {
                'model_size_mb': round(onnx_size_mb, 3),
                'inference_time_ms': round(onnx_ms, 2),
                'accuracy': float(onnx_acc),
            },
            'edge_benchmark': edge,
            'compression': {
                'size_reduction': f"{(1 - onnx_size_mb / pt_size) * 100:.1f}%",
                'speedup_vs_pt': f"{pt_inf / onnx_ms:.1f}x" if onnx_ms > 0 else 'N/A',
            },
        }
        all_results[name] = result

        print(f"  PyTorch:  Acc={pt_metrics['accuracy']:.4f}, Size={pt_size:.3f}MB, Inf={pt_inf:.2f}ms")
        print(f"  ONNX:     Acc={onnx_acc:.4f}, Size={onnx_size_mb:.3f}MB, Inf={onnx_ms:.2f}ms")
        print(f"  Edge:     Mean={edge['mean_ms']:.2f}ms, P50={edge['p50_ms']:.2f}ms, "
              f"P95={edge['p95_ms']:.2f}ms, {edge['samples_per_sec']:.0f} samples/sec")

    results_path = os.path.join(BASE, 'results', 'deployment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
