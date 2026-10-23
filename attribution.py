"""
Split a predicted score into the factors that produced it.

The model sees 41 columns; a reader thinks in five things -- what the market
makes of the player, recent form, whether they play, the underlying numbers, and
the fixture. Each column belongs to exactly one of those, and the prediction is
divided between them with Shapley values computed over the groups rather than
the columns, so the parts are exact and add up:

    base + price + form + minutes + underlying + fixture == predicted points

where `base` is the model's average over its training rows -- the expected score
of a typical player-gameweek, about 1.2 points, most of which are non-players.

Two ways of "removing" a factor were tried, and the choice matters a great deal.

  Interventional (swap the factor for a background player's values) credited
  Haaland's attacking numbers -0.75 points, despite his xG and threat being
  several times the average. Swapping his price for a £5.5m player's while
  keeping his stats builds a player who does not exist, and the trees have
  nothing sensible to say about one.

  Path-dependent (follow the tree where the factor is known, and average over
  both branches by training cover where it is not) keeps every evaluation inside
  the data the model was fitted on. That is what is used here. It is exact for
  tree ensembles and explains all ~670 players in about half a second.

What it explains is the model, not football. Each factor's share is averaged
over every order the factors could be revealed in, so it reflects interactions:
Haaland's form inputs add +0.58 on their own but take away 2.21 once minutes,
underlying numbers, price and fixture are known, and come out at -0.50. A
negative Form is that -- recent returns pulling the number down once the rest
is counted -- not a player out of form.
"""

import math

import numpy as np
import pandas as pd

# Order is the display order everywhere -- legend, stacking and table columns --
# and the report's colours are assigned in it. It is chosen so the stack keeps
# apart the palette's two confusable pairs (slot 2 against slots 4 and 5): slot 3
# is Minutes, the factor least likely to be absent from a top player's bar.
FACTORS = ["Price & ownership", "Form", "Minutes", "Underlying stats", "Fixture"]

FACTOR_NOTES = {
    "Price & ownership": "price and ownership: the market's view of the player",
    "Form": "recent points, bonus and ICT",
    "Minutes": "minutes and starts over recent games",
    "Underlying stats": "xG, xA, threat, creativity, clean sheets, saves",
    "Fixture": "next opponent's difficulty, home or away, blank or double",
}

_MINUTES = {"minutes", "start_rate_5"}
_FORM = {"total_points", "bps", "influence", "ict_index", "pts_per_min"}
# Position is folded in here rather than given its own bar: measured on GW7 of
# 2026-27 it moved no top-20 player by more than 0.06 points.
_PRICE = {"value", "log_selected", "position_id"}
_UNDERLYING = {
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "goals_scored", "assists", "threat", "creativity",
    "clean_sheets", "goals_conceded", "saves",
}


def factor_of(column):
    """
    The factor a model input belongs to.

    Unknown columns raise rather than defaulting, so a new feature has to be
    placed deliberately instead of silently inflating whichever group is last.
    """
    if column.startswith("next_"):
        return "Fixture"
    stat = column.split("_rolling_")[0]
    if stat.endswith("_cumsum"):
        stat = stat[: -len("_cumsum")]
    for factor, stats in (("Minutes", _MINUTES), ("Form", _FORM),
                          ("Price & ownership", _PRICE),
                          ("Underlying stats", _UNDERLYING)):
        if stat in stats:
            return factor
    raise KeyError(f"no factor for model input {column!r}")


def _ensemble(model):
    """(trees, weight per tree) for a supported tree ensemble, else None."""
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        return None
    if hasattr(model, "learning_rate"):
        # Gradient boosting: an (n_stages, 1) array for a single-output regressor.
        trees = [stage[0].tree_ for stage in np.atleast_2d(estimators)]
        return trees, float(model.learning_rate)
    trees = [estimator.tree_ for estimator in estimators]
    return trees, 1.0 / len(trees)


def _unpack(tree):
    return (tree.children_left, tree.children_right, tree.feature, tree.threshold,
            tree.weighted_n_node_samples, tree.value[:, 0, 0], tree.node_count)


def _tree_expectation(arrays, X, known):
    """
    Expected output of one tree when only the `known` columns are observed.

    At a split on a known column the row goes the way its value sends it; at
    one on an unknown column both branches are averaged, weighted by how many
    training rows went each way. sklearn numbers children after their parent, so
    a single reverse pass visits every node after its children.
    """
    left, right, feature, threshold, cover, value, count = arrays
    out = [None] * count
    for node in range(count - 1, -1, -1):
        low, high = left[node], right[node]
        if low == -1:
            out[node] = value[node]
        elif known[feature[node]]:
            out[node] = np.where(X[:, feature[node]] <= threshold[node], out[low], out[high])
        else:
            out[node] = (cover[low] * out[low] + cover[high] * out[high]) / cover[node]
    return np.broadcast_to(out[0], (len(X),))


def explain(model, rows, feature_cols):
    """
    Per-factor contributions to each row's prediction.

    Returns (base, frame): `frame` has one column per entry in FACTORS, indexed
    like `rows`, and base + frame.sum(axis=1) reproduces model.predict(rows).
    Returns None for a model that is not a tree ensemble (the two-stage model
    multiplies its parts, so its prediction is not a sum over trees), or if the
    reconstruction does not match the model's own prediction -- a check that
    catches any change in sklearn's tree internals rather than reporting wrong
    numbers.
    """
    ensemble = _ensemble(model)
    if ensemble is None or rows.empty:
        return None
    trees, weight = ensemble
    arrays = [_unpack(tree) for tree in trees]

    # Trees compare float32 inputs against their thresholds; matching that is
    # what makes the reconstruction exact rather than off by a split.
    X = rows[feature_cols].to_numpy(dtype=np.float32)
    membership = np.array([FACTORS.index(factor_of(c)) for c in feature_cols])

    k = len(FACTORS)
    value = {}
    for mask in range(1 << k):
        known = ((mask >> membership) & 1).astype(bool)
        value[mask] = weight * sum(_tree_expectation(a, X, known) for a in arrays)

    # The ensemble's constant term (the boosting initial estimate; zero for a
    # forest) is whatever separates the trees' sum from the model's prediction.
    predicted = model.predict(rows[feature_cols])
    offset = predicted - value[(1 << k) - 1]
    if np.ptp(offset) > 1e-6:
        return None
    offset = float(offset.mean())

    contributions = np.zeros((len(X), k))
    for i in range(k):
        for mask in range(1 << k):
            if mask >> i & 1:
                continue
            size = bin(mask).count("1")
            share = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            contributions[:, i] += share * (value[mask | 1 << i] - value[mask])

    base = offset + float(value[0][0])
    return base, pd.DataFrame(contributions, columns=FACTORS, index=rows.index)
