import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_iiot_data(n_samples=50000, n_features=65, random_state=42):
    rng = np.random.RandomState(random_state)

    n_normal = int(n_samples * 0.7)
    n_attack = n_samples - n_normal

    normal_data = rng.randn(n_normal, n_features) * 0.5 + 0.1

    attack_patterns = [
        ('DDoS', 1.5, 3.0),
        ('Injection', 2.0, 2.5),
        ('Backdoor', -1.5, 2.0),
        ('MitM', 1.0, -2.0),
        ('Reconnaissance', -2.0, 1.5),
        ('Ransomware', 3.0, -3.0),
    ]

    attack_data = []
    attack_labels = []
    n_per_attack = n_attack // len(attack_patterns)

    for attack_name, offset, scale in attack_patterns:
        attack_chunk = rng.randn(n_per_attack, n_features) * 0.8 + offset
        attack_chunk[:, 0] = attack_chunk[:, 0] * scale
        attack_chunk[:, 5] = attack_chunk[:, 5] + offset * 2
        attack_chunk[:, 10] = attack_chunk[:, 10] - offset
        attack_data.append(attack_chunk)
        attack_labels.extend([attack_name] * n_per_attack)

    remaining = n_attack - len(attack_data) * n_per_attack
    if remaining > 0:
        attack_chunk = rng.randn(remaining, n_features) * 0.8 + 2.0
        attack_data.append(attack_chunk)
        attack_labels.extend(['DDoS'] * remaining)

    attack_data = np.vstack(attack_data)

    X = np.vstack([normal_data, attack_data])
    y_binary = np.array([0] * n_normal + [1] * n_attack)

    indices = rng.permutation(n_samples)
    X = X[indices]
    y_binary = y_binary[indices]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_binary


def generate_synthetic_non_iid_clients(X, y, num_clients=5, random_state=42):
    rng = np.random.RandomState(random_state)
    attack_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    rng.shuffle(attack_idx)
    rng.shuffle(normal_idx)

    attack_splits = np.array_split(attack_idx, num_clients)

    normal_imbalance = np.linspace(0.1, 0.9, num_clients)
    client_data = []

    for i in range(num_clients):
        n_normal_for_client = int(len(normal_idx) * normal_imbalance[i])
        normal_subset = normal_idx[:n_normal_for_client]
        attack_subset = attack_splits[i]
        client_idx = np.concatenate([normal_subset, attack_subset])
        rng.shuffle(client_idx)
        client_data.append((X[client_idx], y[client_idx]))

    return client_data
