# FPL squad optimiser

Picks Fantasy Premier League transfers by predicting next-gameweek points for
every player, then solving for the best legal squad as a mixed-integer program.

## Where it currently stands

Four seasons replayed gameweek by gameweek, each trained only on the season
before it, retraining as it goes and pricing every decision at the gameweek it
was taken (`backtest/validate.py`):

| Season | Hold | Transfers | + chips | Transfer Δ | Chip Δ |
|---|---:|---:|---:|---:|---:|
| 2022-23 | 1742 | 1995 | 2112 | +253 | +117 |
| 2023-24 | 1532 | 1983 | 2075 | +451 | +92 |
| 2024-25 | 1899 | 1956 | 2030 | +57 | +74 |
| 2025-26 | 1806 | 1996 | **2250** | +190 | +254 |

**Both effects replicate: transfers and chips are positive in all four seasons.**
Transfers average **+238** over holding a fixed squad (range +57 to +451), chips a
further **+134** (range +74 to +254). Neither is a single-season fluke.

### But read the totals as percentiles

Raw totals are not comparable across seasons — the median manager scored 2210 in
2022-23 and 2003 in 2025-26. `benchmark.py` samples real managers and places each
total in that season's population:

| Season | median | hold | transfers | + chips |
|---|---:|---:|---:|---:|
| 2022-23 | 2210 | 1742 (29th) | 1995 (53rd) | 2112 (66th) |
| 2023-24 | 2116 | 1532 (14th) | 1983 (54th) | 2075 (64th) |
| 2024-25 | 2153 | 1899 (43rd) | 1956 (48th) | 2030 (55th) |
| 2025-26 | 2003 | 1806 (42nd) | 1996 (68th) | **2250 (96th)** |
| **mean** | | **32nd** | **56th** | **70th** |

**The model is a slightly above-average manager, not a strong one.** It averages
the 70th percentile with chips and the 56th — dead average — without them. The
96th percentile in 2025-26 is an outlier, and the +99 over a human that this
project reported for months came from that one season.

For scale, the human it was compared against finished 2025-26 on 2151, the 88th
percentile. That is better than the model's four-season average.

The likely reason 2025-26 flatters the model: it was the first season with eight
chips instead of five, and the model banked +254 from them against +74 to +117 in
the five-chip seasons. It uses a doubled allocation automatically; humans were
adapting to a rule change. Whether that edge survives contact with 2026-27 is the
open question.

The most encouraging number is the stability of the transfers-only column: 1995,
1983, 1956, 1996 across four seasons — a spread of 40 points. The `hold` baseline
swings by 367 because it depends entirely on which opening squad it happened to
draw, and the transfer engine corrects for that. 2024-25's small +57 is not a bad
year for the model; it is a year when the opening squad was already good.

Chip value is confounded with the allocation: the three five-chip seasons average
+94 against +254 for the single eight-chip season, which cannot be separated from
season luck at n=1.

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
python3 backtest/validate.py             # replicate across four seasons
python3 benchmark.py --backtest          # place those totals in the real population
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
| `chips.py` | chip allocation, valuation and timing |
| `backtest/` | full-season simulation, multi-season replication, worm graph |
| `benchmark.py` | samples real managers to turn totals into percentiles |
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
| Multi-gameweek horizon (plan 3–8 weeks, transfer chain, banked free transfers) | Horizon 3 mean −30, better in 1 of 4 seasons; horizon 5 mean +23, better in 3 of 4 but p≈0.53. Diagnosis below. Available as `--horizon N` in the backtest, default 1. |
| Richer opponent features (attack/defence strength by venue, rolling goals scored and conceded) | No help on either question. Weekly: top-15 paired over 29 folds 4.455 vs 4.577, p=0.27; starters MAE 2.238 → 2.242. Horizon: cross-week variation ratio 0.07 → 0.116, still far short of what planning needs. Behind `features.USE_OPPONENT_FEATURES`. |
| Two-stage model, P(60+ mins) × E[points \| played] | Marginally better at prediction (starters MAE 2.225 vs 2.238, rank ρ 0.363 vs 0.354) but no better at picking squads: top-15 paired over 29 folds 4.623 vs 4.577, p=0.76; backtest 1955 vs 1996. Kept as `--model two-stage` for its calibrated start probability. |

Shrinkage, the evidence gate and the two-stage model all remain available behind
flags (`features.DEFAULT_SHRINK`, `--min-evidence`, `--model two-stage`) and are
off by default.

**Why the horizon failed is the most useful thing here.** The planner works — its
unit tests show it banking transfers and timing a purchase to the week a fixture
run begins. But across a five-week horizon a player's predicted points vary by
0.096 on average against 1.325 of spread *between* players, a ratio of 0.07, and
the mean rank change from week one to week five is 43 places out of 780. The five
gameweeks are very nearly the same ranking problem repeated, so there is nothing
to plan around.

The cause is upstream of the optimiser. Form is held constant across the horizon
by construction, so the only varying inputs are the three next-fixture columns,
which together carry under a tenth of the model's feature importance. **The
blocker for multi-week planning is fixture sensitivity in the model, not the
solver** — which moves "better information" from last place to first on the
roadmap.

The lesson underneath: lightly-evidenced cheap players are not bad buys, because
they free budget for premiums. And four changes to the *estimator or the solver*
moved squad quality by nothing, against one missing capability (chips) worth more
than the entire gap to a human.

## Chips

Eight chips a season — two each of wildcard, free hit, bench boost and triple
captain, one of each per half. Windows are read from the live API, since FPL
doubled the allocation in 2025-26 and hardcoding the old rules silently wastes
half of them.

Each chip is valued in expected points and compared against a bar that decays to
zero at the end of its window, so a chip is spent early only on a standout week
but is never left to expire. Deferring everything to the deadline scores badly
(+4 on 2025-26): chips collide at the window's final gameweek and only one can be
played, so the rest are lost.

`main.py` reports which chips are available and whether each clears its bar.
Because the API does not expose a manager's remaining chips, pass the ones you
have already used: `--chips-used TC BB`.

## Known gaps

- **The horizon is one gameweek.** Serious solvers plan 5–8 with decaying weights.
- **Budget uses current price, not selling price**, overstating funds on risen
  players. The public API does not expose selling price.
- **No historical injury data**, so the availability gate is live-only and the
  backtest can field players who were actually ruled out.
- **Free transfers default to 1**; the API does not expose your real count.

## Roadmap

Performance work looks finished. Six attempts to improve the model or the solver
have now returned nothing, against one missing capability (chips) worth more than
the entire gap to a human. Fixture detail was the last mechanism with a clear
story behind it, and richer opponent data moved neither the weekly pick nor the
cross-week variation that multi-week planning needs. What remains is correctness
and delivery:

1. **Bounded correctness fixes** — selling price (budget currently overstates
   funds on risen players), the vice-captain (not modelled at all), and
   `--free-transfers` defaulting to 1 when you may have banked up to 5.
2. **Make it get used** — a scheduled weekly job that posts the recommendation,
   rather than something to remember to run.
3. **Separate chip allocation from season luck** — the eight-chip season is a
   sample of one, so the +254 cannot yet be attributed to the doubled allocation.
   Next season's data settles it at no cost.
