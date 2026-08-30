"""
Points prediction model.

Training deliberately does not split internally. Splitting inside `train_model` is
how the old version ended up reporting a random-split score on panel data, where
adjacent gameweeks of the same player land on both sides of the split and their
rolling windows overlap. Splitting is the caller's job, and `evaluate.py` does it
on time.
"""

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

import features

# Re-exported so callers have one import for the pipeline.
prepare = features.prepare
feature_columns = features.feature_columns
latest_rows = features.latest_rows

# 'two-stage' is handled in train_model rather than here, since it builds several
# estimators; suffix it (e.g. 'two-stage:rf') to change the underlying regressor.
SELECTABLE = ["gbr", "rf", "two-stage"]

MODELS = {
    "gbr": lambda: GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
    "rf": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ),
}

FULL_GAME_MINUTES = 60  # the FPL threshold for two appearance points and clean sheets


class TwoStageModel:
    """
    Separate "will they play" from "what will they score if they do".

    A single regressor over the whole panel spends most of its capacity learning
    that non-players score nothing -- roughly 60% of rows -- and `minutes_rolling_3`
    dominates its feature importance as a result. Splitting the question lets each
    part be modelled on its own terms:

        P(60+ mins)  x  E[points | 60+ mins]
      + P(cameo)     x  E[points | cameo]

    where a cameo is any appearance under 60 minutes, worth one appearance point
    plus whatever they return. Expected points is the probability-weighted sum, so
    a player who is excellent but rotated is correctly marked down without the
    regressor having to learn rotation itself.

    Measured, and not the default. It is marginally better at prediction --
    starters MAE 2.225 vs 2.238, rank correlation 0.363 vs 0.354 -- but that does
    not reach squad quality: top-15 per gameweek paired over 29 folds gives 4.623
    vs 4.577, better in 14/29, p=0.76, and the full-season backtest came out at
    1955 against the single regressor's 1996. Available via `--model two-stage`.

    It is still worth keeping for something the single regressor cannot express:
    `clf_start` is a calibrated probability that a player starts, which is
    directly readable as rotation risk.
    """

    def __init__(self, kind="gbr"):
        self.kind = kind
        self.clf_start = None    # P(minutes >= 60)
        self.clf_appear = None   # P(minutes > 0)
        self.reg_start = None    # E[points | minutes >= 60]
        self.reg_cameo = None    # E[points | 0 < minutes < 60]
        self.cameo_fallback = 1.0

    def _classifier(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    def fit(self, X, y_points, y_minutes):
        started = y_minutes >= FULL_GAME_MINUTES
        appeared = y_minutes > 0
        cameo = appeared & ~started

        self.clf_start = self._fit_classifier(X, started)
        self.clf_appear = self._fit_classifier(X, appeared)

        self.reg_start = build_model(self.kind)
        if started.sum() >= 50:
            self.reg_start.fit(X[started], y_points[started])
        else:
            self.reg_start.fit(X, y_points)

        if cameo.sum() >= 50:
            self.reg_cameo = build_model(self.kind)
            self.reg_cameo.fit(X[cameo], y_points[cameo])
        else:
            self.cameo_fallback = float(y_points[cameo].mean()) if cameo.any() else 1.0

        return self

    def _fit_classifier(self, X, y):
        """A classifier needs both classes; degenerate targets fall back to a constant."""
        if y.nunique() < 2:
            return float(y.iloc[0]) if len(y) else 0.0
        return self._classifier().fit(X, y)

    @staticmethod
    def _probability(model, X):
        if isinstance(model, float):
            return np.full(len(X), model)
        return model.predict_proba(X)[:, 1]

    def predict(self, X):
        p_start = self._probability(self.clf_start, X)
        p_appear = self._probability(self.clf_appear, X)
        # An appearance probability below the start probability is incoherent;
        # clamp so the cameo weight cannot go negative.
        p_cameo = np.clip(p_appear - p_start, 0.0, 1.0)

        points_start = self.reg_start.predict(X)
        if self.reg_cameo is not None:
            points_cameo = self.reg_cameo.predict(X)
        else:
            points_cameo = np.full(len(X), self.cameo_fallback)

        return p_start * points_start + p_cameo * points_cameo

    @property
    def feature_importances_(self):
        """Importances of the points-given-started regressor, the interpretable part."""
        return getattr(self.reg_start, "feature_importances_", None)


DEFAULT_MODEL = "gbr"


def build_model(kind=DEFAULT_MODEL):
    if kind not in MODELS:
        raise ValueError(f"unknown model {kind!r}, expected one of {sorted(MODELS)}")
    return MODELS[kind]()


def train_model(train_df, feature_cols, kind=DEFAULT_MODEL):
    """
    Fit on an already-selected training set.

    `kind` of 'two-stage' builds the minutes/points decomposition; anything else
    is a single regressor over raw next-gameweek points.
    """
    X = train_df[feature_cols]
    if kind.startswith("two-stage"):
        base = kind.split(":", 1)[1] if ":" in kind else "gbr"
        return TwoStageModel(kind=base).fit(
            X, train_df["target_points"], train_df["target_minutes"]
        )

    model = build_model(kind)
    model.fit(X, train_df["target_points"])
    return model


def predict(model, df, feature_cols):
    """Predict points for rows that have complete features."""
    usable = df.dropna(subset=feature_cols)
    out = df.copy()
    out["predicted_points"] = 0.0
    if len(usable):
        out.loc[usable.index, "predicted_points"] = model.predict(usable[feature_cols])
    return out


def feature_importance(model, feature_cols, top=15):
    """Feature importances, most important first."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = sorted(zip(feature_cols, importances), key=lambda p: -p[1])
    return pairs[:top]
