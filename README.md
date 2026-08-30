# FPL squad optimiser

Predicts gameweek points and picks transfers with a mixed-integer program.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./setup_data.sh          # historical data, not tracked
```

`setup_data.sh` sparse-clones only the seasons the model reads (~118MB, ~20s)
rather than the full 333MB archive going back to 2016-17. Pass season labels to
override the default three.

The vendored dataset lags the live season, usually by a gameweek. Finished
gameweeks missing from it are pulled from the API automatically, and the tool
warns when the form it is using is older than the gameweek it is planning for.
Refresh with `git -C data/historical pull`.

## Use

```bash
python3 main.py --team-id 8936155        # transfer recommendations
python3 main.py --wildcard               # rebuild from scratch
python3 evaluate.py                      # walk-forward model evaluation
python3 backtest/backtest.py             # full-season simulation
```

## Layout

| File | Role |
|---|---|
| `identity.py` | stable player/team identity across seasons |
| `data_fetcher.py` | cached FPL API client, availability flags |
| `features.py` | player-gameweek panel and feature construction |
| `ml_model.py` | model definitions and fitting |
| `evaluate.py` | walk-forward evaluation against naive baselines |
| `optimizer.py` | squad selection MILP |
| `main.py` | weekly recommendation CLI |
| `backtest/` | full-season simulation and worm graph |

## Things worth knowing

**Player identity is `code`, not `element`.** FPL reassigns `element` ids every
season — 462 of the 467 players present in both 2025-26 and 2026-27 changed id.
Joining across seasons on `element` attaches every prediction to the wrong player
and fails silently. Everything upstream of the optimizer keys on `code`; ids are
resolved only where we talk to the API.

**Read the evaluation, not the MAE.** About 60% of player-gameweek rows are
players who did not play, so predicting zero for everyone gets an MAE around 1.12.
`evaluate.py` therefore prints naive baselines beside the model and segments by
whether the player was likely to start. On the metric that matters — the points
actually scored by the highest-predicted players — the model returns 6.00 per
player against 4.26 for the best naive predictor.

**The backtest prices decisions at the gameweek they were taken.** Using today's
prices lets the simulation buy a player at a price they only reached in April.

## Things that were tried and did not work

Recorded so they are not re-attempted. Early in a season the tool recommends
players it has seen once, on the strength of a single big score. That looks
obviously wrong. Three fixes were built and measured, and none of them helped.

| Attempt | Result |
|---|---|
| Hierarchical shrinkage of form toward career, then positional, means | No effect. Top-15 per gameweek, paired over 35 folds: 4.676 → 4.703, better in 16/35, p=0.86, 95% CI [−0.27, +0.32]. MAE ~1% worse in every evidence segment. |
| `log_career_gws` as a model feature | −62 points over the 2025-26 backtest (1996 → 1934). |
| Minimum-evidence gate on transfers in | Worse at every threshold: 1934 ungated, 1865 at 5, 1921 at 10 and 20. |

Both the shrinkage and the gate remain available behind flags
(`features.DEFAULT_SHRINK`, `--min-evidence`) and are off by default.

Two lessons worth keeping. A pooled top-50 across folds suggested shrinkage was
a large win; pairing fold by fold showed it was noise, so prefer the paired test.
And cheap, lightly-evidenced players are not obviously bad buys — they free
budget for premiums, and blocking them cost points.

**Known gaps.** Budget uses current price rather than selling price, which
overstates funds on risen players (the public API does not expose selling price).
The backtest has no historical injury data, so the availability gate is live-only.
Chips are not played in the backtest. The horizon is one gameweek.
