import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from src.features.wav2vec import load_audio, pick_device

TARGET_SR = 16000

def speed_perturb(a, factor):
    idx = np.arange(0, len(a), factor)
    return np.interp(idx, np.arange(len(a)), a).astype("float32")

def add_noise(a, snr_db=20, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    rms = np.sqrt(np.mean(a ** 2) + 1e-9)
    return (a + rng.normal(0, rms / (10 ** (snr_db / 20)), len(a))).astype("float32")

# Saca el embedding en las layers pedidas, promediando en frames
def emb(a, fe, model, layers, device=None):
    device = pick_device(device)
    iv = fe(a, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        hs = model(iv, output_hidden_states=True).hidden_states
    return torch.cat([hs[l].mean(1).squeeze(0) for l in layers], dim=-1).cpu().numpy().astype("float32")

def extract_augmented(metadata, model_name="microsoft/wavlm-large", layers=(13, 11, 10), device=None, seed=42):
    device = pick_device(device)
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    rng = np.random.default_rng(seed)

    # para cada audio, se calcula el embedding del audio original, una version mas rapida, una mas lenta y otra con ruido
    Xc, Xa, ya, ga = [], [], [], []
    for _, row in metadata.iterrows():
        a = load_audio(row["path"])
        Xc.append(emb(a, fe, model, layers, device))
        for aug in (speed_perturb(a, 0.9), speed_perturb(a, 1.1), add_noise(a, 20, rng)):
            Xa.append(emb(aug, fe, model, layers, device)); ya.append(row["emotion"]); ga.append(row["actor"])
    return (np.array(Xc, "float32"), np.array(Xa, "float32"), np.array(ya), np.array(ga))
