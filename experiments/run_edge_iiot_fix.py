import sys, os, json, time
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
from utils.metrics import evaluate_model, compute_model_size_mb, count_parameters, measure_inference_time, estimate_flops, train_one_epoch

device = 'cpu'
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_file = os.path.join(base, 'results', 'real_data_results.json')
f = os.path.join(base, 'data', 'Edge-IIoTset', 'kaggle', 'DNN-EdgeIIoT-dataset.csv')
lc = 'Attack_type'

print("Reading DNN-EdgeIIoT-dataset.csv in chunks for stratified sampling...")
sample_size = 20000
chunks = pd.read_csv(f, chunksize=100000, low_memory=False)

col_names = None
normal_samples = []
attack_samples = []

for i, chunk in enumerate(chunks):
    if col_names is None:
        col_names = list(chunk.columns)
        drop_cols = ['frame.time', 'Attack_type', 'binary_label']
        feature_cols = [c for c in col_names if c not in drop_cols]
        print(f"Features: {len(feature_cols)}")

    chunk['binary_label'] = chunk[lc].apply(lambda x: 0 if str(x).strip() == 'Normal' else 1)

    normal = chunk[chunk['binary_label'] == 0]
    attack = chunk[chunk['binary_label'] == 1]
    normal_samples.append(normal)
    attack_samples.append(attack)

    if (i + 1) % 5 == 0:
        print(f"  Chunk {i+1}: total normal={sum(len(n) for n in normal_samples):,}, attack={sum(len(a) for a in attack_samples):,}")

normal_all = pd.concat(normal_samples, ignore_index=True)
attack_all = pd.concat(attack_samples, ignore_index=True)
print(f"Total: Normal={len(normal_all):,}, Attack={len(attack_all):,}")

# Stratified 20K sample
n_normal = min(len(normal_all), 14000)
n_attack = min(len(attack_all), 6000)
normal_sampled = normal_all.sample(n=n_normal, random_state=42)
attack_sampled = attack_all.sample(n=n_attack, random_state=42)
df_sample = pd.concat([normal_sampled, attack_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Sample: {len(df_sample)} rows, attack ratio: {df_sample['binary_label'].mean():.2%}")

# Preprocess: encode all categorical columns
for c in feature_cols:
    if df_sample[c].dtype == 'object':
        df_sample[c] = LabelEncoder().fit_transform(df_sample[c].astype(str))

X = df_sample[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
y = df_sample['binary_label'].values.astype(np.int64)
scaler = StandardScaler()
X = scaler.fit_transform(X)
input_dim = X.shape[1]
print(f"Input dim: {input_dim}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train
configs = [
    ('CNN-BiLSTM (Baseline)', CNNBiLSTM(input_dim, 2)),
    ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, 2)),
    ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, 2, pruning_threshold=0.02)),
]

results = {}
for mname, model in configs:
    t0 = time.time()
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"\n>>> {mname}")
    for ep in range(10):
        loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 128, device)
        if (ep + 1) % 5 == 0:
            vm = evaluate_model(model, X_val, y_val, device)
            print(f"  Epoch {ep+1}/10 - Loss: {loss:.4f}, Val Acc: {vm['accuracy']:.4f}")
    me = evaluate_model(model, X_test, y_test, device)
    sz = compute_model_size_mb(model)
    pa = count_parameters(model)
    ti = measure_inference_time(model, X_test, device)
    fl = estimate_flops(model, input_dim, device)
    sp = 0.0
    if hasattr(model, 'apply_pruning'):
        model.apply_pruning()
        sp = model.get_sparsity()
    elapsed = time.time() - t0
    print(f"  Acc={me['accuracy']:.4f} F1={me['f1_score']:.4f} Size={sz:.3f}MB Params={pa:,} Inf={ti:.3f}ms FLOPs={fl:.2f}M Sparsity={sp:.2%} [{elapsed:.0f}s]")
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

# Save
edge_key = 'Edge-IIoTset-DNN'
edge_res = {edge_key: {'input_dim': input_dim, 'models': results}}
with open(results_file) as f:
    all_results = json.load(f)
all_results.update(edge_res)
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nEdge-IIoTset-DNN results saved.")
b = list(results.values())[0]
for mn, m in results.items():
    short = mn.split(' (')[0]
    sz_red = (1 - m['model_size_mb']/b['model_size_mb'])*100
    sp_up = b['inference_time_ms']/m['inference_time_ms']
    print(f"  {short:<20} Acc={m['accuracy']:.4f} F1={m['f1_score']:.4f} "
          f"Size={m['model_size_mb']:.3f}MB Params={m['num_parameters']:,} "
          f"Inf={m['inference_time_ms']:.2f}ms FLOPs={m['flops_millions']:.2f}M"
          + (f" Sparsity={m['sparsity']:.1%}" if m['sparsity'] > 0 else "")
          + (f" ({sz_red:+.1f}%, {sp_up:.2f}x)" if 'Baseline' not in mn else ""))
