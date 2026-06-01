import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunedLightweightCNNGRU(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=64, num_layers=1,
                 dropout=0.2, pruning_threshold=0.01):
        super(PrunedLightweightCNNGRU, self).__init__()
        self.input_dim = input_dim
        self.pruning_threshold = pruning_threshold

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        context = gru_out[:, -1, :]
        x = F.relu(self.bn4(self.fc1(context)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def apply_pruning(self):
        masks = {}
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask = (param.abs() > self.pruning_threshold).float()
                    param.data.mul_(mask)
                    masks[name] = mask
        return masks

    def get_sparsity(self):
        total = 0
        nonzero = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                total += param.numel()
                nonzero += (param.abs() > self.pruning_threshold).sum().item()
        if total == 0:
            return 0.0
        return 1.0 - (nonzero / total)
