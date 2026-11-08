from collections import defaultdict

import pandas as pd

FEATURE_COLUMNS = [
	"spread_line",
	"total_line",
	"home_implied_probability",
	"away_implied_probability",
	"elo_difference",
	"home_coach_win_rate",
	"away_coach_win_rate",
	"home_coach_games",
	"away_coach_games",
]


def _implied_probability(odds):
	if pd.isna(odds) or odds == 0:
		return None
	odds = float(odds)
	if odds < 0:
		return -odds / (-odds + 100)
	return 100 / (odds + 100)


def build_features(games):
	"""Build pregame features in chronological order without using current results."""
	ordered = games.copy()
	if "gameday" in ordered:
		ordered["_sort_date"] = pd.to_datetime(ordered["gameday"], errors="coerce")
		sort_columns = ["_sort_date", *(column for column in ("season", "week") if column in ordered)]
		ordered = ordered.sort_values(sort_columns, kind="stable")
	elif {"season", "week"}.issubset(ordered.columns):
		ordered = ordered.sort_values(["season", "week"], kind="stable")

	team_elo = defaultdict(lambda: 1500.0)
	coach_wins = defaultdict(float)
	coach_games = defaultdict(int)
	feature_rows = {}

	for index, game in ordered.iterrows():
		home_team = game.get("home_team")
		away_team = game.get("away_team")
		home_coach = game.get("home_coach")
		away_coach = game.get("away_coach")
		home_coach = home_coach if pd.notna(home_coach) else None
		away_coach = away_coach if pd.notna(away_coach) else None
		home_odds = _implied_probability(game.get("home_moneyline"))
		away_odds = _implied_probability(game.get("away_moneyline"))
		both_odds = home_odds is not None and away_odds is not None
		if both_odds:
			odds_total = home_odds + away_odds

		feature_rows[index] = {
			"spread_line": game.get("spread_line"),
			"total_line": game.get("total_line"),
			"home_implied_probability": home_odds / odds_total if both_odds else 0.5,
			"away_implied_probability": away_odds / odds_total if both_odds else 0.5,
			"elo_difference": team_elo[home_team] - team_elo[away_team],
			"home_coach_win_rate": (
				coach_wins[home_coach] / coach_games[home_coach]
				if home_coach and coach_games[home_coach]
				else 0.5
			),
			"away_coach_win_rate": (
				coach_wins[away_coach] / coach_games[away_coach]
				if away_coach and coach_games[away_coach]
				else 0.5
			),
			"home_coach_games": coach_games[home_coach] if home_coach else 0,
			"away_coach_games": coach_games[away_coach] if away_coach else 0,
		}

		home_score = game.get("home_score")
		away_score = game.get("away_score")
		if pd.notna(home_score) and pd.notna(away_score) and pd.notna(home_team) and pd.notna(away_team):
			home_win = float(home_score > away_score)
			expected_home = 1 / (1 + 10 ** ((team_elo[away_team] - team_elo[home_team]) / 400))
			team_elo[home_team] += 20 * (home_win - expected_home)
			team_elo[away_team] += 20 * ((1 - home_win) - (1 - expected_home))
			for coach, result in ((home_coach, home_win), (away_coach, 1 - home_win)):
				if coach:
					coach_wins[coach] += result
					coach_games[coach] += 1

	return pd.DataFrame.from_dict(feature_rows, orient="index").reindex(games.index)[FEATURE_COLUMNS].fillna(0.0)
