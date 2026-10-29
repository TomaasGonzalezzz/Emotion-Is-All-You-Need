from pathlib import Path

import pandas as pd

from src import config

def parse_filename(path: Path) -> dict:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Nombre RAVDESS inesperado: {path.name}")
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    actor_int = int(actor)
    return {
        "path": str(path),
        "modality": config.MODALITY_MAP[modality],
        "vocal_channel": config.VOCAL_CHANNEL_MAP[vocal],
        "emotion": config.EMOTION_MAP[emotion],
        "intensity": config.INTENSITY_MAP[intensity],
        "statement": config.STATEMENT_MAP[statement],
        "repetition": int(repetition),
        "actor": actor_int,
        "gender": config.actor_gender(actor_int),
    }

def build_metadata() -> pd.DataFrame:
    rows = []
    for base in (config.SPEECH_DIR, config.SONG_DIR):
        for wav in sorted(base.rglob("*.wav")):
            rows.append(parse_filename(wav))
    return pd.DataFrame(rows)

# De un df, devolvemos el mismo pero sin las columnas de metadata del RAVDESS
def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in config.META_COLS]

# Filtrar por canal (speech/song) y split si se pide devolver (X = features, y = emoción, groups = actor)
def get_xy(df: pd.DataFrame, vocal_channel: str = "speech", split: str | None = None):
    sub = df
    if vocal_channel is not None:
        sub = sub[sub["vocal_channel"] == vocal_channel]
    if split is not None:
        sub = sub[sub["split"] == split]
    feat_cols = feature_columns(sub)
    return sub[feat_cols], sub["emotion"], sub["actor"]
