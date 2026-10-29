import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel

TARGET_SR = 16000

def pick_device(device=None):
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio), sr, TARGET_SR
        ).numpy()
    return audio

def extract_wav2vec(metadata: pd.DataFrame, model_name: str = "facebook/wav2vec2-base", device: str = None, layer: int = 8) -> pd.DataFrame:
    device = pick_device(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    paths = metadata["path"].tolist()
    embeddings = []
    
    # para cada audio se extraen las features (1, T, 768) donde T es la cantidad de frames por feature que se arman (1 cada 20ms)
    # despues nos quedamos con una capa en particular y promediamos sobre el eje de los T frames y squeeze para tener un vector con 768 features
    with torch.no_grad():
        for p in paths:
            audio = load_audio(p)
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            hidden = model(input_values, output_hidden_states=True).hidden_states[layer]
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)

    arr = np.vstack(embeddings).astype("float32")
    cols = [f"wav2vec_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=pd.Index(paths, name="path"), columns=cols)

# arma un dataframe a partir de las capas que se soliciten de un output de wav2vec
def layers_to_dataframe(emb, paths, layers) -> pd.DataFrame:
    sel = emb[:, layers, :].reshape(emb.shape[0], -1)
    cols = [f"wav2vec_l{l}_{i}" for l in layers for i in range(emb.shape[2])]
    return pd.DataFrame(sel.astype("float32"), index=pd.Index(paths, name="path"), columns=cols)

def extract_wav2vec_layers(metadata: pd.DataFrame, model_name: str = "facebook/wav2vec2-base", device: str = None):
    device = pick_device(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    paths = metadata["path"].tolist()
    out = []
    
    # mas o menos lo mismo que extract_wav2vec pero aca no se queda con una capa en particular, terminas con un vector de n_capas * 768 features
    with torch.no_grad():
        for p in paths:
            audio = load_audio(p)
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            hidden_states = model(input_values, output_hidden_states=True).hidden_states
            layer_means = torch.stack([h.mean(dim=1).squeeze(0) for h in hidden_states], dim=0)
            out.append(layer_means.cpu().numpy())

    emb = np.stack(out, axis=0).astype("float32")
    return emb, paths

def extract_wav2vec_frames(metadata: pd.DataFrame, model_name: str = "facebook/wav2vec2-base", device: str = None, layers=(8,)):
    device = pick_device(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    paths = metadata["path"].tolist()
    seqs = []
    
    # Hacemos mas o menos lo mismo solo que no promediamos sobre los frames. Termina quedando algo de dimension (T_i, len(layers)*768)
    # Importante que no todos los T son iguales para cada audio, depende de su duracion, las layers se concatenan en un vector de features, para cada frame hay len(layers)*768 features
    with torch.no_grad():
        for p in paths:
            audio = load_audio(p)
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            hs = model(input_values, output_hidden_states=True).hidden_states
            sel = torch.cat([hs[l].squeeze(0) for l in layers], dim=-1)
            seqs.append(sel.cpu().numpy().astype("float32"))
    return seqs, paths
