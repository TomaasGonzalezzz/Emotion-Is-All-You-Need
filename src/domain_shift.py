import re

import pandas as pd

EMOTION_MAP_EMODB = {
    "W": "angry", "L": "boredom", "E": "disgust",
    "A": "fearful", "F": "happy", "T": "sad", "N": "neutral",
}
SHARED_EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad"]

EMODB_FILENAME_RE = re.compile(r"(\d{2})([a-z]\d{2})([WLEAFTN])([a-z])\.wav")


def parse_emodb_filename(path):
    m = EMODB_FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"no matchea el patron de nombre EMODB esperado: {path.name}")
    actor, _, emo_code, _ = m.groups()
    return {"path": str(path), "actor": actor, "emotion": EMOTION_MAP_EMODB[emo_code]}


def build_emodb_metadata(root):
    rows = [parse_emodb_filename(p) for p in sorted(root.glob("*.wav"))]
    return pd.DataFrame(rows)
