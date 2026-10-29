import opensmile
import pandas as pd

def extract_egemaps(metadata: pd.DataFrame) -> pd.DataFrame:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    paths = metadata["path"].tolist()
    feats = smile.process_files(paths)
    feats.index = feats.index.get_level_values("file")
    feats.index.name = "path"
    return feats.loc[paths]
