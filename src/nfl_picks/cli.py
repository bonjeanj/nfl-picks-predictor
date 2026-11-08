import json

import joblib
import pandas as pd
import typer

from nfl_picks.config import BIAS_PATH, GAMES_PATH, MODEL_PATH
from nfl_picks.data.fetch_games import fetch_games
from nfl_picks.features.build_features import FEATURE_COLUMNS, build_features
from nfl_picks.models.predict import predict
from nfl_picks.models.train import train_model

app = typer.Typer()


def _load_games(path):
	if path.exists():
		return pd.read_csv(path)
	return fetch_games()


def _load_biases(path):
	if not path.exists():
		return {}, {}
	with path.open(encoding="utf-8") as file:
		values = json.load(file)
	return values.get("teams", {}), values.get("coaches", {})


@app.command()
def fetch():
	"""Download the latest game schedule and historical results."""
	GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
	games = fetch_games()
	games.to_csv(GAMES_PATH, index=False)
	typer.echo(f"Saved {len(games)} games to {GAMES_PATH}")


@app.command()
def train():
	"""Train the winner model using available historical games and betting lines."""
	games = _load_games(GAMES_PATH)
	features = build_features(games)
	completed = games["home_score"].notna() & games["away_score"].notna()
	targets = (games.loc[completed, "home_score"] > games.loc[completed, "away_score"]).astype(int)
	if targets.nunique() < 2:
		raise typer.BadParameter("Training data must include wins and losses for the home team.")

	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	model = train_model(features.loc[completed, FEATURE_COLUMNS], targets)
	joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
	typer.echo(f"Trained on {len(targets)} completed games; model saved to {MODEL_PATH}")


@app.command()
def predict_week(week: int = typer.Option(..., help="NFL week to predict."), season: int = typer.Option(None, help="Season year; defaults to the latest in the data.")):
	"""Pick winners for scheduled matchups in a given week."""
	if not MODEL_PATH.exists():
		raise typer.BadParameter("No trained model found. Run `nfl-picks train` first.")

	games = _load_games(GAMES_PATH)
	current_season = season if season is not None else int(games["season"].max())
	upcoming = games[
		(games["season"] == current_season)
		& (games["week"] == week)
		& games["home_score"].isna()
		& games["away_score"].isna()
	]
	if upcoming.empty:
		typer.echo(f"No unplayed matchups found for {current_season}, week {week}.")
		raise typer.Exit()

	features = build_features(games).loc[upcoming.index]
	artifact = joblib.load(MODEL_PATH)
	team_biases, coach_biases = _load_biases(BIAS_PATH)
	probabilities = predict(
		artifact["model"],
		features[artifact["feature_columns"]],
		upcoming,
		team_biases,
		coach_biases,
	)
	for (_, matchup), probability in zip(upcoming.iterrows(), probabilities):
		winner = matchup["home_team"] if probability >= 0.5 else matchup["away_team"]
		confidence = probability if probability >= 0.5 else 1 - probability
		typer.echo(
			f"{matchup['away_team']} at {matchup['home_team']}: {winner} ({confidence:.1%})"
		)


if __name__ == "__main__":
	app()
