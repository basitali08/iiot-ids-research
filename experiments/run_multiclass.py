import sys, os, json, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from models.pruned_lightweight_cnn_gru import PrunedLightweightCNNGRU
from utils.metrics import evaluate_model, compute_model_size_mb, count_parameters, measure_inference_time, estimate_flops, train_one_epoch

device = 'cpu'
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

f = os.path.join(base, 'data', 'Edge-IIoTset', 'kaggle', 'DNN-EdgeIIoT-dataset.csv')
lc = 'Attack_type'

print("Loading Edge-IIoTset in chunks for multi-class...")
chunks = pd.read_csv(f, chunksize=100000, low_memory=False)
samples_by_class = {}
for i, chunk in enumerate(chunks):
    for label, group in chunk.groupby(lc):
        if label not in samples_by_class:
            samples_by_class[label] = []
        samples_by_class[label].append(group)
    if (i + 1) % 5 == 0:
        total = sum(sum(len(df) for df in dfs) for dfs in samples_by_class.values())
        print(f"  Chunk {i+1}: {total:,} rows accumulated")

# Concatenate per class
class_dfs = {}
for label, dfs in samples_by_class.items():
    class_dfs[label] = pd.concat(dfs, ignore_index=True)
    print(f"  {label:<25} {len(class_dfs[label]):>8,}")

# Sample: up to 2000 per class, min 5000 total
sample_target = min(2000, min(len(df) for df in class_dfs.values()))
sampled = []
for label, df in class_dfs.items():
    n = min(len(df), sample_target)
    sampled.append(df.sample(n=n, random_state=42))

df_sample = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nTotal sample: {len(df_sample)} rows")

# Create label encoding
label_encoder = LabelEncoder()
df_sample['label_id'] = label_encoder.fit_transform(df_sample[lc])
num_classes = len(label_encoder.classes_)
print(f"Classes: {num_classes}")
for i, cls in enumerate(label_encoder.classes_):
    n = (df_sample['label_id'] == i).sum()
    print(f"  {i}: {cls:<25} {n}")

# Preprocess features
drop_cols = ['frame.time', lc, 'label_id']
feature_cols = [c for c in df_sample.columns if c not in drop_cols]

for c in feature_cols:
    if df_sample[c].dtype == 'object':
        df_sample[c] = LabelEncoder().fit_transform(df_sample[c].astype(str))

X = df_sample[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
y = df_sample['label_id'].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X)
input_dim = X.shape[1]
print(f"Features: {input_dim}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train models (multi-class = num_classes)
configs = [
    ('CNN-BiLSTM', CNNBiLSTM(input_dim, num_classes)),
    ('CNN-GRU (Lightweight)', LightweightCNNGRU(input_dim, num_classes)),
    ('Pruned CNN-GRU', PrunedLightweightCNNGRU(input_dim, num_classes, pruning_threshold=0.02)),
]

all_results = {}
for mname, model in configs:
    t0 = time.time()
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"\n>>> {mname}")
    for ep in range(12):
        loss = train_one_epoch(model, X_train, y_train, optimizer, criterion, 128, device)
        if (ep + 1) % 4 == 0:
            vm = evaluate_model(model, X_val, y_val, device, average='macro')
            print(f"  Epoch {ep+1}/12 - Loss: {loss:.4f}, Val Acc: {vm['accuracy']:.4f}")

    # Test
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    accuracy = (preds == y_test).mean()
    report = classification_report(y_test, preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    sz = compute_model_size_mb(model)
    pa = count_parameters(model)
    ti = measure_inference_time(model, X_test, device)
    fl = estimate_flops(model, input_dim, device)
    sp = 0.0
    if hasattr(model, 'apply_pruning'):
        model.apply_pruning()
        sp = model.get_sparsity()
    elapsed = time.time() - t0

    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    print(f"  Acc={accuracy:.4f} Macro-F1={macro_f1:.4f} W-F1={weighted_f1:.4f}")
    print(f"  Size={sz:.3f}MB Params={pa:,} Inf={ti:.3f}ms FLOPs={fl:.2f}M" +
          (f" Sparsity={sp:.2%}" if sp else "") + f" [{elapsed:.0f}s]")

    all_results[mname] = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'model_size_mb': float(sz),
        'num_parameters': int(pa),
        'inference_time_ms': float(ti),
        'flops_millions': float(fl),
        'sparsity': float(sp),
        'per_class': {cls: report[cls] for cls in label_encoder.classes_ if cls in report},
    }

outfile = os.path.join(base, 'results', 'multiclass_results.json')
with open(outfile, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*60}")
print("MULTI-CLASS SUMMARY (Edge-IIoTset, 15 classes)")
print(f"{'='*60}")
for mn, m in all_results.items():
    print(f"\n  {mn}:")
    print(f"    Accuracy: {m['accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}, Weighted-F1: {m['weighted_f1']:.4f}")
    print(f"    Size: {m['model_size_mb']:.3f}MB, Params: {m['num_parameters']:,}, Inf: {m['inference_time_ms']:.2f}ms")
    print(f"    Per-class F1:")
    for cls, cm in sorted(m['per_class'].items()):
        print(f"      {cls:<25} {cm.get('f1-score', 0):.4f}")

print(f"\nSaved to {outfile}")
