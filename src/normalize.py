import numpy as np
import pandas as pd

# Resta la media y divide por la varianza de cada feature separado por hablante
# Terminamos teniendo cuanto se desvia una persona de su voz base
def cmvn_by_speaker(X: pd.DataFrame, groups) -> pd.DataFrame:
    groups = pd.Series(np.asarray(groups), index=X.index)
    mu = X.groupby(groups).transform("mean")
    sd = X.groupby(groups).transform("std", ddof=0).replace(0, 1.0)
    return (X - mu) / sd

# lo mismo que arriba pero contemplando que hay T frames
def cmvn_frames_by_speaker(seqs, groups):
    groups = np.asarray(groups)
    out = [None] * len(seqs)
    for spk in np.unique(groups):
        idx = np.where(groups == spk)[0]
        stacked = np.concatenate([np.asarray(seqs[i], dtype="float32") for i in idx], axis=0)
        mu = stacked.mean(axis=0)
        sd = stacked.std(axis=0)
        sd[sd == 0] = 1.0
        for i in idx:
            out[i] = ((np.asarray(seqs[i], dtype="float32") - mu) / sd).astype("float32")
    return out
