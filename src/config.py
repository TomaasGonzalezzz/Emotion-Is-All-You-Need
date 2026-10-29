from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
SPEECH_DIR = DATA_DIR / "Audio_Speech_Actors_01-24"
SONG_DIR = DATA_DIR / "Audio_Song_Actors_01-24"
PROCESSED_DIR = DATA_DIR / "processed"
EMODB_DIR = DATA_DIR / "emodb_raw" / "extracted" / "wav"
SPLITS_DIR = PROJECT_ROOT / "splits"
SPLIT_PATH = SPLITS_DIR / "split.json"

RANDOM_SEED = 42

WAV2VEC_TOP_LAYERS = [8, 6, 7]

MODALITY_MAP = {"01": "full-AV", "02": "video-only", "03": "audio-only"}
VOCAL_CHANNEL_MAP = {"01": "speech", "02": "song"}
EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
INTENSITY_MAP = {"01": "normal", "02": "strong"}
STATEMENT_MAP = {"01": "kids", "02": "dogs"}


def actor_gender(actor: int) -> str:
    return "male" if actor % 2 == 1 else "female"


META_COLS = [
    "path", "modality", "vocal_channel", "emotion", "intensity",
    "statement", "repetition", "actor", "gender", "split",
]
