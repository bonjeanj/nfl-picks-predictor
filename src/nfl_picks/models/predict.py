import numpy as np


def predict(model, X, matchups=None, team_biases=None, coach_biases=None):
	"""Return home-win probabilities with optional user log-odds adjustments."""
	probabilities = model.predict_proba(X)[:, 1]
	if matchups is None:
		return probabilities

	team_biases = team_biases or {}
	coach_biases = coach_biases or {}
	adjusted = []
	for probability, (_, matchup) in zip(probabilities, matchups.iterrows()):
		bounded = np.clip(probability, 1e-6, 1 - 1e-6)
		log_odds = np.log(bounded / (1 - bounded))
		adjustment = (
			team_biases.get(matchup.get("home_team"), 0.0)
			- team_biases.get(matchup.get("away_team"), 0.0)
			+ coach_biases.get(matchup.get("home_coach"), 0.0)
			- coach_biases.get(matchup.get("away_coach"), 0.0)
		)
		adjusted.append(1 / (1 + np.exp(-(log_odds + adjustment))))
	return np.asarray(adjusted)
