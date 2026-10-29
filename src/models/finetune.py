import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification, get_linear_schedule_with_warmup)
from src import config
from src.features.wav2vec import load_audio

TARGET_SR = 16000

def pick_device(device=None):
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def augment_audio(a, rng):
    f = rng.choice([0.9, 1.0, 1.1])
    if f != 1.0:
        idx = np.arange(0, len(a), f)
        a = np.interp(idx, np.arange(len(a)), a).astype("float32")
    if rng.random() < 0.5:
        rms = np.sqrt(np.mean(a ** 2) + 1e-9)
        a = (a + rng.normal(0, rms / 10.0, len(a))).astype("float32")
    return a

class _AudioDataset(Dataset):
    def __init__(self, audios, y, augment=False, seed=0):
        self.audios, self.y, self.augment = audios, y, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        a = self.audios[i]
        if self.augment:
            a = augment_audio(a, self.rng)
        return a, self.y[i]

class FineTuneClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model_name="facebook/wav2vec2-base", lr=2e-5, weight_decay=1e-4, warmup_ratio=0.1, batch_size=4, max_epochs=30, patience=6, val_frac=0.15, freeze_feature_encoder=True, freeze_layers=0, augment=False, max_audio_len=88000, class_weight="balanced", verbose=False, seed=config.RANDOM_SEED, device=None):
        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.freeze_feature_encoder = freeze_feature_encoder
        self.freeze_layers = freeze_layers
        self.augment = augment
        self.max_audio_len = max_audio_len
        self.class_weight = class_weight
        self.verbose = verbose
        self.seed = seed
        self.device = device

    def collate(self, batch):
        audios, ys = zip(*batch)
        inputs = self.proc_(list(audios), sampling_rate=TARGET_SR, return_tensors="pt", padding="max_length", max_length=self.max_audio_len, truncation=True, return_attention_mask=True)
        return inputs, torch.tensor(np.asarray(ys))

    def loader(self, audios, yi, shuffle, augment=False):
        return DataLoader(_AudioDataset(audios, yi, augment=augment, seed=self.seed), batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        dev = pick_device(self.device)

        audios = [load_audio(p) for p in X]
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y)))
        idx_of = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx_of[v] for v in y])

        tr, va = train_test_split(np.arange(len(audios)), test_size=self.val_frac, random_state=self.seed, stratify=yi)

        self.proc_ = AutoFeatureExtractor.from_pretrained(self.model_name)
        tr_loader = self.loader([audios[i] for i in tr], yi[tr], shuffle=True, augment=self.augment)
        va_loader = self.loader([audios[i] for i in va], yi[va], shuffle=False)

        weight = None
        if self.class_weight == "balanced":
            w = compute_class_weight("balanced", classes=np.arange(len(self.classes_)), y=yi[tr])
            weight = torch.tensor(w, dtype=torch.float32, device=dev)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.model_ = AutoModelForAudioClassification.from_pretrained(
            self.model_name, num_labels=len(self.classes_)).to(dev)
        if self.freeze_feature_encoder:
            self.model_.freeze_feature_encoder()
        if self.freeze_layers > 0:
            base = self.model_.base_model
            for p in base.feature_projection.parameters():
                p.requires_grad = False
            for layer in base.encoder.layers[:self.freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False
        params = [p for p in self.model_.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.max_epochs * len(tr_loader)
        scheduler = get_linear_schedule_with_warmup(
            opt, int(self.warmup_ratio * total_steps), total_steps)

        best_acc, best_state, bad = -1.0, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.model_.train()
            for inputs, yb in tr_loader:
                opt.zero_grad()
                logits = self.model_(**inputs.to(dev)).logits
                loss = loss_fn(logits, yb.to(dev))
                loss.backward()
                opt.step()
                scheduler.step()
            self.model_.eval()
            correct = total = 0
            with torch.no_grad():
                for inputs, yb in va_loader:
                    pred = self.model_(**inputs.to(dev)).logits.argmax(1).cpu()
                    correct += (pred == yb).sum().item()
                    total += len(yb)
            val_acc = correct / total
            self.history_.append({"epoch": epoch, "val_acc": val_acc})
            if self.verbose:
                print(f"[finetune] epoch {epoch} val_acc {val_acc:.3f}", flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        self.best_val_acc_ = best_acc
        return self

    def predict(self, X):
        dev = pick_device(self.device)
        audios = [load_audio(p) for p in X]
        loader_ = self.loader(audios, np.zeros(len(audios), dtype=int), shuffle=False)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for inputs, _ in loader_:
                preds.append(self.model_(**inputs.to(dev)).logits.argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(preds)]
