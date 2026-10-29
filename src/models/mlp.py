import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from src import config

def pick_device(device=None):
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class MLPNet(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(in_dim)]
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(512, 256), dropout=0.3, lr=1e-3, weight_decay=1e-4, batch_size=64, max_epochs=100, patience=10, val_frac=0.15, class_weight="balanced", seed=config.RANDOM_SEED, device=None):
        self.hidden = hidden
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

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = pick_device(self.device)

        # armamos etiquetas
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        idx_of = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx_of[v] for v in y])

        # split estratificado por clase
        Xtr, Xval, ytr, yval = train_test_split(X, yi, test_size=self.val_frac, random_state=self.seed, stratify=yi)

        # aplicamos escalado, el val se escala con los valores fiteados de train para evitar leakage
        self.scaler_ = StandardScaler().fit(Xtr)
        Xtr, Xval = self.scaler_.transform(Xtr), self.scaler_.transform(Xval)

        # caso de rebalanceo de clases (util para darle mas peso a la clase neutral minoritaria)
        weight = None
        if self.class_weight == "balanced":
            w = compute_class_weight("balanced", classes=np.arange(len(self.classes_)), y=ytr)
            weight = torch.tensor(w, dtype=torch.float32, device=dev)

        # armamos red con adam y demas
        self.net_ = MLPNet(X.shape[1], len(self.classes_), tuple(self.hidden), self.dropout).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        # config para usar bien el m2 en train
        Xtr_t = torch.tensor(Xtr, device=dev)
        ytr_t = torch.tensor(ytr, device=dev)
        Xval_t = torch.tensor(Xval, device=dev)
        yval_t = torch.tensor(yval, device=dev)

        best_acc, best_state, bad = -1.0, None, 0
        self.history_ = []
        n = len(Xtr_t)
        for epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(n, device=dev)
            running = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()
                running += loss.item() * len(idx)
            self.net_.eval()
            with torch.no_grad():
                val_acc = (self.net_(Xval_t).argmax(1) == yval_t).float().mean().item()
            self.history_.append({"epoch": epoch, "train_loss": running / n, "val_acc": val_acc})

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
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_proba(self, X):
        dev = pick_device(self.device)
        X = self.scaler_.transform(np.asarray(X, dtype="float32"))
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(X, device=dev))
            return torch.softmax(logits, dim=1).cpu().numpy()
