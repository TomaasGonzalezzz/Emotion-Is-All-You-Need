import warnings

import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoFeatureExtractor, AutoModel, logging as hf_logging

warnings.filterwarnings("ignore", message="X does not have valid feature names")
hf_logging.set_verbosity_error()

from src import config
from src.features.wav2vec import load_audio, pick_device
from src.normalize import cmvn_by_speaker

BACKBONES = {"wavlm": "microsoft/wavlm-large", "hubert": "facebook/hubert-large-ll60k", "base": "facebook/wav2vec2-base"}
TARGET_SR = 16000

class EmotionPredictor:

    def __init__(self, model_path=None, device=None):
        self.m = joblib.load(model_path or (config.PROCESSED_DIR / "final_model.pkl"))
        self.m["mlp_aug"].device = "cpu"
        self.device = pick_device(device)
        self._fe, self._net = {}, {}

    def backbone(self, key):
        if key not in self._net:
            name = BACKBONES[key]
            self._fe[key] = AutoFeatureExtractor.from_pretrained(name)
            self._net[key] = AutoModel.from_pretrained(name).to(self.device).eval()
        return self._fe[key], self._net[key]

    def embed(self, key, audios, layers):
        fe, net = self.backbone(key)
        out = []
        with torch.no_grad():
            for a in audios:
                iv = fe(a, sampling_rate=TARGET_SR, return_tensors="pt").input_values.to(self.device)
                hs = net(iv, output_hidden_states=True).hidden_states
                out.append(torch.cat([hs[l].mean(1).squeeze(0) for l in layers], dim=-1).cpu().numpy())
        return pd.DataFrame(np.array(out, dtype="float32"))

    def predict_proba(self, audio_paths, speakers=None):
        audios = [load_audio(p) for p in audio_paths]
        L = self.m["layers"]
        Xw = self.embed("wavlm", audios, L["wavlm"])
        Xh = self.embed("hubert", audios, L["hubert"])
        Xb = self.embed("base", audios, L["base"])

        def norm(X):
            if speakers is None:
                return X
            return cmvn_by_speaker(X.reset_index(drop=True), np.asarray(speakers))

        w = self.m["weights"]
        P = (w["wavlm"]  * self.m["lr_wavlm"].predict_proba(norm(Xw)) +
             w["hubert"] * self.m["lr_hubert"].predict_proba(norm(Xh)) +
             w["base"]   * self.m["lr_base"].predict_proba(norm(Xb)) +
             w["mlp"]    * self.m["mlp_aug"].predict_proba(norm(pd.DataFrame(np.hstack([Xw.values, Xb.values])))))
        return P, self.m["lr_wavlm"].classes_

    def predict(self, audio_paths, speakers=None):
        P, classes = self.predict_proba(audio_paths, speakers)
        return [classes[i] for i in P.argmax(axis=1)]

def predict(audio_paths, speakers=None, model_path=None):
    return EmotionPredictor(model_path).predict(audio_paths, speakers)

if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    if not paths:
        print("uso: python -m src.predict audio1.wav [audio2.wav ...]")
        sys.exit(1)
    for p, emo in zip(paths, EmotionPredictor().predict(paths)):
        print(f"{emo:10s}  {p}")
