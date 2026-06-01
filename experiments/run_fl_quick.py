import sys, os, copy, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

from models.baseline_cnn_bilstm import CNNBiLSTM
from models.lightweight_cnn_gru import LightweightCNNGRU
from utils.synthetic_data import generate_synthetic_iiot_data
from utils.data_loader import create_federated_clients
from utils.metrics import evaluate_model, train_one_epoch, federated_averaging

device = 'cpu'
print("Generating data...")
X, y = generate_synthetic_iiot_data(n_samples=5000, n_features=65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

results = {}
for non_iid_mode, mode_name in [(False, 'IID'), (True, 'Non-IID')]:
    print(f"\n=== FL {mode_name} ===")
    client_data = create_federated_clients(X_train, y_train, num_clients=3, non_iid=non_iid_mode)

    for model_name, ModelClass in [
        ("CNN-BiLSTM", CNNBiLSTM),
        ("CNN-GRU", LightweightCNNGRU)
    ]:
        print(f"\n--- {model_name} ---")
        model = ModelClass(65, 2)
        global_model = copy.deepcopy(model)
        client_models = [copy.deepcopy(global_model) for _ in range(3)]
        criterion = nn.CrossEntropyLoss()
        round_accs = []

        for rnd in range(8):
            for ci, (Xc, yc) in enumerate(client_data):
                cm = client_models[ci]
                opt = torch.optim.Adam(cm.parameters(), lr=0.001)
                for _ in range(2):
                    train_one_epoch(cm, Xc, yc, opt, criterion, 64, device)

            global_model = federated_averaging(global_model, client_models)
            for cm in client_models:
                cm.load_state_dict(global_model.state_dict())

            m = evaluate_model(global_model, X_test, y_test, device)
            round_accs.append(float(m["accuracy"]))
            print(f"  Round {rnd+1}: Acc={m['accuracy']:.4f}")

        results[f"{model_name}_{mode_name}"] = round_accs

print("\n" + "="*50)
print("FL CONVERGENCE COMPARISON")
print("="*50)
for key, accs in results.items():
    print(f"{key:<25} Final Acc: {accs[-1]:.4f}")

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'fl_results.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out}")
