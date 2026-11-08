import pandas as pd
URL="https://github.com/nflverse/nfldata/releases/download/games/games.csv"
def fetch_games(): return pd.read_csv(URL)
