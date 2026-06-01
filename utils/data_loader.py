import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess_edge_iiotset(filepath, sample_frac=1.0, random_state=42):
    cols = [
        'frame.time', 'ip.id', 'ip.src_host', 'ip.dst_host', 'ip.proto',
        'ip.len', 'tcp.srcport', 'tcp.dstport', 'tcp.seq', 'tcp.ack',
        'tcp.len', 'tcp.ack_raw', 'tcp.flags', 'tcp.window_size',
        'udp.srcport', 'udp.dstport', 'udp.len', 'udp.window_size',
        'http.request.uri', 'http.request.method', 'http.file_raw_data',
        'http.response', 'http.status.code', 'http.request.version',
        'http.Content_type', 'http.Server', 'http.Host',
        'http.Accept_Encoding', 'http.Connection', 'http.Content_Length',
        'http.User_Agent', 'http.Content_Type', 'http.Content_Encoding',
        'icmp.type', 'icmp.code', 'icmp.seq_le', 'icmp.seq_ge',
        'mqtt.conack.flags', 'mqtt.conack.val', 'mqtt.protoname',
        'mqtt.topic', 'mqtt.msg', 'mqtt.qos', 'mqtt.ver',
        'coap.id', 'coap.uri_path', 'coap.type', 'coap.code',
        'coap.opts', 'coap.msgid', 'coap.status_code',
        'dns.id', 'dns.flags', 'dns.qry.name', 'dns.qry.type',
        'dns.qry.class', 'dns.qry.len', 'dns.ra',
        'arp.opcode', 'arp.src.hw_mac', 'arp.src.proto_ipv4',
        'arp.dst.hw_mac', 'arp.dst.proto_ipv4', 'arp.hw.len',
        'arp.hw.size', 'arp.hw.type', 'arp.proto.len', 'arp.proto.size',
        'data.binary', 'label', 'type'
    ]

    df = pd.read_csv(filepath, low_memory=False)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'class', 'attack_type', 'type']:
            label_col = col
            break

    if label_col is None:
        label_col = df.columns[-1]

    df['binary_label'] = df[label_col].apply(
        lambda x: 1 if str(x).lower() not in ['0', 'normal', 'benign', 'normal traffic'] else 0
    )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['binary_label', label_col]]

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [label_col]]

    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df['binary_label'].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


def load_and_preprocess_xiiotid(filepath, sample_frac=1.0, random_state=42):
    df = pd.read_csv(filepath, low_memory=False)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'class', 'attack', 'binary_label']:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]

    df['binary_label'] = df[label_col].apply(
        lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1
    )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['binary_label', label_col]]

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [label_col]]

    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['binary_label'].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


def load_and_preprocess_wustl(filepath, sample_frac=1.0, random_state=42):
    df = pd.read_csv(filepath, low_memory=False)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    label_col = None
    for col in df.columns:
        if col.lower() in ['label', 'class', 'attack', 'binary_label', 'target']:
            label_col = col
            break
    if label_col is None:
        label_col = df.columns[-1]

    df['binary_label'] = df[label_col].apply(
        lambda x: 0 if str(x).lower() in ['0', 'normal', 'benign'] else 1
    )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['binary_label', label_col]]

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [label_col]]

    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].values.astype(np.float32)
    y = df['binary_label'].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_federated_clients(X, y, num_clients=5, non_iid=False, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)

    if non_iid:
        attack_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        rng.shuffle(attack_idx)
        rng.shuffle(normal_idx)

        client_data = []
        for i in range(num_clients):
            if i < num_clients - 1:
                a_size = len(attack_idx) // num_clients
                n_size = len(normal_idx) // num_clients
            else:
                a_size = len(attack_idx) - (num_clients - 1) * (len(attack_idx) // num_clients)
                n_size = len(normal_idx) - (num_clients - 1) * (len(normal_idx) // num_clients)

            client_attack = attack_idx[i * (len(attack_idx) // num_clients):
                                       i * (len(attack_idx) // num_clients) + a_size]
            client_normal = normal_idx[i * (len(normal_idx) // num_clients):
                                       i * (len(normal_idx) // num_clients) + n_size]
            client_idx = np.concatenate([client_attack, client_normal])
            rng.shuffle(client_idx)
            client_data.append((X[client_idx], y[client_idx]))
    else:
        rng.shuffle(indices)
        chunk_size = n_samples // num_clients
        client_data = []
        for i in range(num_clients):
            start = i * chunk_size
            end = start + chunk_size if i < num_clients - 1 else n_samples
            client_data.append((X[indices[start:end]], y[indices[start:end]]))

    return client_data
