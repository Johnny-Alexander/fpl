# FPL squad optimiser

Picks Fantasy Premier League transfers by predicting next-gameweek points for
every player, then solving for the best legal squad as a mixed-integer program.

## Where it currently stands

Replaying the whole of 2025-26 gameweek by gameweek, retraining as it goes and
pricing every decision at the gameweek it was taken:

| | Season total |
|---|---:|
| Human manager (this account's real 2025-26) | **2151** |
| Model, weekly transfers | 1996 |
| Model squad, never transferred again | 1806 |

Two things to read from that. The transfer engine is worth **+190** over holding
a fixed squad, so the machinery does something. And the model still **trails a
competent human by 155 points**, so it is not yet worth following blindly.

The model plays no chips in that simulation; the human total includes theirs.

## Setup

Python 3.9+.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./setup_data.sh
```

`setup_data.sh` sparse-clones only the seasons the model reads from
[vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
— about 118MB and 20 seconds, against 333MB for the full archive back to 2016-17.
Pass season labels to override the default three.

That dataset is a third-party mirror and usually lags the live season by a
gameweek. Finished gameweeks missing from it are pulled from the API
automatically, and the tool warns when the form it is using is older than the
gameweek it is planning for.

## Use

```bash
python3 main.py                          # transfer recommendations for your team
python3 main.py --team-id 1234567        # someone else's team
python3 main.py --wildcard               # rebuild the squad from scratch
python3 main.py --free-transfers 2       # if you have banked transfers

python3 evaluate.py                      # walk-forward evaluation vs naive baselines
python3 backtest/backtest.py             # full-season simulation + worm graph
```

Your team id is in the URL when you are logged in:
`fantasy.premierleague.com/entry/<ID>/event/3`.

## How it fits together

| File | Role |
|---|---|
| `identity.py` | stable player and team identity across seasons |
| `data_fetcher.py` | cached FPL API client, availability flags |
| `features.py` | player-gameweek panel and feature construction |
| `ml_model.py` | model definitions and fitting |
| `evaluate.py` | walk-forward evaluation against naive baselines |
| `optimizer.py` | squad selection MILP |
| `main.py` | weekly recommendation CLI |
| `backtest/` | full-season simulation and worm graph |
| `setup_data.sh` | fetches the historical dataset |

Data flows one way: raw gameweek CSVs → dense player-gameweek panel keyed on
stable `code` → features → model → predicted points → MILP → squad. Season-local
FPL ids only exist at the two ends, where the API is read and where a
recommendation is printed.

## Things worth knowing

**Player identity is `code`, not `element`.** FPL reassigns `element` ids every
season — 462 of the 467 players present in both 2025-26 and 2026-27 changed id.
Joining across seasons on `element` attaches every prediction to the wrong player
and fails silently, producing confident recommendations about the wrong people.

**Read the evaluation, not the MAE.** About 60% of player-gameweek rows are
players who did not play, so predicting zero for everyone scores an MAE around
1.12 and looks respectable. `evaluate.py` prints naive baselines beside the model
and segments by whether a player was likely to start.

**Judge changes by paired per-gameweek tests.** A pooled top-50 across folds once
suggested a large win that vanished under pairing. The metric that decides is the
realised points of the top-15 predicted players, per gameweek, paired between
arms across folds.

**The panel is dense.** Double gameweeks are collapsed to one row and blank
gameweeks are explicit zero rows, so `shift(-1)` genuinely means "next gameweek".
Without that, doubles are mislabelled and blanks are skipped entirely.

**The backtest prices decisions at the gameweek they were taken.** Using today's
prices lets the simulation buy a player at a price they only reached in April.

## Things that were tried and did not work

Recorded so they are not re-attempted. Early in a season the tool recommends
players it has seen once, on the strength of a single big score. That looks
obviously wrong, and fixing it did not help.

| Attempt | Result |
|---|---|
| Hierarchical shrinkage of form toward career, then positional, means | No effect. Top-15 per gameweek, paired over 35 folds: 4.676 → 4.703, better in 16/35, p=0.86, 95% CI [−0.27, +0.32]. MAE ~1% worse in every evidence segment. |
| `log_career_gws` as a model feature | −62 points over the 2025-26 backtest (1996 → 1934). |
| Minimum-evidence gate on transfers in | Worse at every threshold: 1934 ungated, 1865 at 5, 1921 at 10 and 20. |
| Two-stage model, P(60+ mins) × E[points \| played] | Marginally better at prediction (starters MAE 2.225 vs 2.238, rank ρ 0.363 vs 0.354) but no better at picking squads: top-15 paired over 29 folds 4.623 vs 4.577, p=0.76; backtest 1955 vs 1996. Kept as `--model two-stage` for its calibrated start probability. |

Shrinkage, the evidence gate and the two-stage model all remain available behind
flags (`features.DEFAULT_SHRINK`, `--min-evidence`, `--model two-stage`) and are
off by default.

The lesson underneath: lightly-evidenced cheap players are not bad buys, because
they free budget for premiums. And three changes to the *estimator* moved squad
quality by nothing, which suggests the estimator is not the binding constraint.

## Known gaps

- **Chips are never played** in the backtest, while the human total includes them.
- **The horizon is one gameweek.** Serious solvers plan 5–8 with decaying weights.
- **Budget uses current price, not selling price**, overstating funds on risen
  players. The public API does not expose selling price.
- **No historical injury data**, so the availability gate is live-only and the
  backtest can field players who were actually ruled out.
- **Free transfers default to 1**; the API does not expose your real count.

## Roadmap

1. **Chips** — play wildcard, free hit, bench boost and triple captain. Missing
   functionality rather than a modelling problem; the optimizer already accepts
   `chip_active`.
2. **Multi-gameweek horizon** — the largest structural gap. The MILP has the right
   shape and needs a gameweek index on the decision variables plus a transfer-state
   chain between them.
3. **Better information** — opponent defensive strength rather than FPL's 1–5
   difficulty, per-90 rates, set-piece and penalty duty.
