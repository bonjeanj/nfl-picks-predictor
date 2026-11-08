# NFL Picks

An NFL winner predictor that learns from historical results, betting lines, team Elo, and each coach's pregame win history. Personal preferences are applied as small log-odds adjustments after the model prediction, so they can be changed without retraining.

## Run It

```powershell
nfl-picks fetch
nfl-picks train
nfl-picks predict-week --week 10
```

`fetch` downloads nflverse game data to `data/games.csv`. Refresh it before training or making weekly picks. `train` saves the model to `models/nfl_model.joblib`. You can also edit `data/games.csv` to provide updated odds or coach names; the feature builder uses `spread_line`, `total_line`, `home_moneyline`, `away_moneyline`, `home_coach`, and `away_coach` when present. Missing line or coach values receive neutral defaults.

The week defaults to the latest season in the data. To pick an earlier season, use `nfl-picks predict-week --week 10 --season 2025`.

## Personal Preferences

Create `config/personal_biases.json` with team abbreviations and coach names as they appear in `data/games.csv`:

```json
{
	"teams": {
		"BUF": 0.2,
		"NE": -0.1
	},
	"coaches": {
		"Example Coach": 0.15
	}
}
```

Values are additive log-odds, not percentages or points. Positive values favor that team or coach; negative values count against them. Small values such as `0.1` to `0.3` provide a modest nudge. These are personal overrides, not learned model weights.
