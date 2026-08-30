"""
Points prediction model.

Training deliberately does not split internally. Splitting inside `train_model` is
how the old version ended up reporting a random-split score on panel data, where
adjacent gameweeks of the same player land on both sides of the split and their
rolling windows overlap. Splitting is the caller's job, and `evaluate.py` does it
on time.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import features

# Re-exported so callers have one import for the pipeline.
prepare = features.prepare
feature_columns = features.feature_columns
latest_rows = features.latest_rows

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

DEFAULT_MODEL = "gbr"


def build_model(kind=DEFAULT_MODEL):
    if kind not in MODELS:
        raise ValueError(f"unknown model {kind!r}, expected one of {sorted(MODELS)}")
    return MODELS[kind]()


def train_model(train_df, feature_cols, kind=DEFAULT_MODEL):
    """Fit on an already-selected training set."""
    model = build_model(kind)
    model.fit(train_df[feature_cols], train_df["target_points"])
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
    if not hasattr(model, "feature_importances_"):
        return []
    pairs = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda p: -p[1]
    )
    return pairs[:top]
