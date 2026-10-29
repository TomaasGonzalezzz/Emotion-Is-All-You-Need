import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

def make_split(
    metadata: pd.DataFrame,
    val_actor_fraction: float = 0.2,
    seed: int = config.RANDOM_SEED,
    out_path: Path = config.SPLIT_PATH,
) -> dict:
    actors = metadata[["actor", "gender"]].drop_duplicates()
    rng = np.random.default_rng(seed)

    val_actors = []
    for gender, grp in actors.groupby("gender"):
        ids = np.sort(grp["actor"].to_numpy())
        n_val = round(len(ids) * val_actor_fraction)
        chosen = rng.choice(ids, size=n_val, replace=False)
        val_actors.extend(int(a) for a in chosen)

    val_set = set(val_actors)
    train_actors = [int(a) for a in sorted(actors["actor"]) if a not in val_set]

    split = {
        "train_actors": sorted(train_actors),
        "val_actors": sorted(val_actors),
        "seed": seed,
        "val_actor_fraction": val_actor_fraction,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2))
    return split

def load_split(path: Path = config.SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text())

def assign_split(metadata: pd.DataFrame, split: dict) -> pd.DataFrame:
    val_set = set(split["val_actors"])
    out = metadata.copy()
    out["split"] = np.where(out["actor"].isin(val_set), "val", "train")
    return out
