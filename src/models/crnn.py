import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src import config

def pick_device(device=None):
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class CRNNNet(nn.Module):
    def __init__(self, in_dim, n_classes, conv_channels=256, rnn_hidden=128, kernel=5, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, conv_channels, kernel_size=kernel, padding=kernel // 2),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(conv_channels, rnn_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * rnn_hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * rnn_hidden, n_classes)

    def forward(self, x, lengths):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None].to(out.device)
        score = self.attn(out).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(score, dim=1).unsqueeze(-1)
        ctx = (out * w).sum(dim=1)
        return self.head(self.drop(ctx))

class _SeqDataset(Dataset):
    def __init__(self, seqs, y):
        self.seqs, self.y = seqs, y

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], self.y[i]

def collate(batch):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    feat = np.asarray(seqs[0]).shape[1]
    padded = torch.zeros(len(seqs), int(lengths.max()), feat, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.from_numpy(np.asarray(s, dtype="float32"))
    return padded, lengths, torch.tensor(np.asarray(ys))

class CRNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, conv_channels=256, rnn_hidden=128, kernel=5, dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=32, max_epochs=60, patience=8, val_frac=0.15, class_weight="balanced", seed=config.RANDOM_SEED, device=None):
        self.conv_channels = conv_channels
        self.rnn_hidden = rnn_hidden
        self.kernel = kernel
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.class_weight = class_weight
        self.seed = seed
        self.device = device

    def loader(self, seqs, yi, shuffle):
        return DataLoader(_SeqDataset(seqs, yi), batch_size=self.batch_size,
                          shuffle=shuffle, collate_fn=collate)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = pick_device(self.device)

        seqs = list(X)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        idx_of = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx_of[v] for v in y])

        tr, va = train_test_split(np.arange(len(seqs)), test_size=self.val_frac,
                                  random_state=self.seed, stratify=yi)
        tr_loader = self.loader([seqs[i] for i in tr], yi[tr], shuffle=True)
        va_loader = self.loader([seqs[i] for i in va], yi[va], shuffle=False)

        weight = None
        if self.class_weight == "balanced":
            w = compute_class_weight("balanced", classes=np.arange(len(self.classes_)), y=yi[tr])
            weight = torch.tensor(w, dtype=torch.float32, device=dev)

        feat = np.asarray(seqs[0]).shape[1]
        self.net_ = CRNNNet(feat, len(self.classes_), self.conv_channels,
                            self.rnn_hidden, self.kernel, self.dropout).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        best_acc, best_state, bad = -1.0, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.net_.train()
            for padded, lengths, yb in tr_loader:
                opt.zero_grad()
                loss = loss_fn(self.net_(padded.to(dev), lengths), yb.to(dev))
                loss.backward()
                opt.step()
            self.net_.eval()
            correct = total = 0
            with torch.no_grad():
                for padded, lengths, yb in va_loader:
                    pred = self.net_(padded.to(dev), lengths).argmax(1).cpu()
                    correct += (pred == yb).sum().item()
                    total += len(yb)
            val_acc = correct / total
            self.history_.append({"epoch": epoch, "val_acc": val_acc})
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.net_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.net_.load_state_dict(best_state)
        self.best_val_acc_ = best_acc
        return self

    def predict(self, X):
        dev = pick_device(self.device)
        seqs = list(X)
        loader_ = self.loader(seqs, np.zeros(len(seqs), dtype=int), shuffle=False)
        self.net_.eval()
        preds = []
        with torch.no_grad():
            for padded, lengths, _ in loader_:
                preds.append(self.net_(padded.to(dev), lengths).argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(preds)]
