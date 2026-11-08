from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
GAMES_PATH = DATA / "games.csv"
MODEL_PATH = ROOT / "models" / "nfl_model.joblib"
BIAS_PATH = ROOT / "config" / "personal_biases.json"
