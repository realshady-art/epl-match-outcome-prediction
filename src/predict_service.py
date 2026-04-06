from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.cache_manager import CacheManager
from src.config import CACHE_TTL_HOURS, OUTPUT_MODELS_DIR, TARGET_SEASON
from src.features import get_feature_columns
from src.live_data_provider import FootballDataCsvProvider, LiveDataProvider, MatchupContext


def _model_path(model_name: str = "logistic_regression") -> Path:
    return Path(OUTPUT_MODELS_DIR) / f"{model_name}.joblib"


def _build_live_feature_row(
    fixture: MatchupContext,
    home_data: dict[str, Any],
    away_data: dict[str, Any],
    h2h_data: dict[str, Any],
) -> pd.DataFrame:
    row = {
        "home_rank_before_match": fixture.home_rank,
        "away_rank_before_match": fixture.away_rank,
        "rank_diff": fixture.away_rank - fixture.home_rank,
        "home_rest_days": home_data["rest_days"],
        "away_rest_days": away_data["rest_days"],
        "rest_days_diff": home_data["rest_days"] - away_data["rest_days"],
        "match_month": fixture.comparison_month,
        "head_to_head_home_points_last_3": h2h_data["home_points_last_3"],
        "head_to_head_away_points_last_3": h2h_data["away_points_last_3"],
        "head_to_head_home_goal_diff_last_3": h2h_data["home_goal_diff_last_3"],
        "home_points_last_3": home_data["points_last_3"],
        "away_points_last_3": away_data["points_last_3"],
        "home_goals_for_last_3": home_data["goals_for_last_3"],
        "away_goals_for_last_3": away_data["goals_for_last_3"],
        "home_goals_against_last_3": home_data["goals_against_last_3"],
        "away_goals_against_last_3": away_data["goals_against_last_3"],
        "home_goal_diff_last_3": home_data["goals_for_last_3"] - home_data["goals_against_last_3"],
        "away_goal_diff_last_3": away_data["goals_for_last_3"] - away_data["goals_against_last_3"],
        "home_win_rate_last_3": home_data["wins_last_3"] / 3.0 if home_data["wins_last_3"] + home_data["draws_last_3"] + home_data["losses_last_3"] else 0.0,
        "away_win_rate_last_3": away_data["wins_last_3"] / 3.0 if away_data["wins_last_3"] + away_data["draws_last_3"] + away_data["losses_last_3"] else 0.0,
        "points_diff_last_3": home_data["points_last_3"] - away_data["points_last_3"],
        "goal_diff_form_last_3": (
            (home_data["goals_for_last_3"] - home_data["goals_against_last_3"])
            - (away_data["goals_for_last_3"] - away_data["goals_against_last_3"])
        ),
        "home_points_last_5": home_data["points_last_5"],
        "away_points_last_5": away_data["points_last_5"],
        "home_goals_for_last_5": home_data["goals_for_last_5"],
        "away_goals_for_last_5": away_data["goals_for_last_5"],
        "home_goals_against_last_5": home_data["goals_against_last_5"],
        "away_goals_against_last_5": away_data["goals_against_last_5"],
        "home_goal_diff_last_5": home_data["goals_for_last_5"] - home_data["goals_against_last_5"],
        "away_goal_diff_last_5": away_data["goals_for_last_5"] - away_data["goals_against_last_5"],
        "home_win_rate_last_5": home_data["wins_last_5"] / 5.0,
        "away_win_rate_last_5": away_data["wins_last_5"] / 5.0,
        "points_diff_last_5": home_data["points_last_5"] - away_data["points_last_5"],
        "goal_diff_form_last_5": (
            (home_data["goals_for_last_5"] - home_data["goals_against_last_5"])
            - (away_data["goals_for_last_5"] - away_data["goals_against_last_5"])
        ),
    }
    return pd.DataFrame([row])[get_feature_columns()]


def predict_match(
    home_team: str,
    away_team: str,
    home_snapshot: str = "now",
    away_snapshot: str = "now",
    provider: LiveDataProvider | None = None,
    cache: CacheManager | None = None,
    model_name: str = "logistic_regression",
) -> dict[str, Any]:
    provider = provider or FootballDataCsvProvider()
    cache = cache or CacheManager()

    fixture_key = f"{home_team}__{home_snapshot}__{away_team}__{away_snapshot}"
    fixture_data = cache.load("fixture", fixture_key, CACHE_TTL_HOURS["fixture"])
    if fixture_data is None:
        fixture = provider.get_matchup_context(home_team, away_team, home_snapshot=home_snapshot, away_snapshot=away_snapshot)
        fixture_data = asdict(fixture)
        cache.store("fixture", fixture_key, fixture_data)
    fixture = MatchupContext(**fixture_data)

    home_key = f"{fixture.home_team_id}__{fixture.home_snapshot_label}"
    away_key = f"{fixture.away_team_id}__{fixture.away_snapshot_label}"
    h2h_key = (
        f"{fixture.home_team_id}__{fixture.home_snapshot_label}__"
        f"{fixture.away_team_id}__{fixture.away_snapshot_label}"
    )

    home_data = cache.load("team_form", home_key, CACHE_TTL_HOURS["team_form"])
    if home_data is None:
        home_data = asdict(
            provider.get_team_recent_form(
                fixture.home_team,
                fixture.home_team_id,
                home_snapshot,
            )
        )
        cache.store("team_form", home_key, home_data)

    away_data = cache.load("team_form", away_key, CACHE_TTL_HOURS["team_form"])
    if away_data is None:
        away_data = asdict(
            provider.get_team_recent_form(
                fixture.away_team,
                fixture.away_team_id,
                away_snapshot,
            )
        )
        cache.store("team_form", away_key, away_data)

    h2h_data = cache.load("head_to_head", h2h_key, CACHE_TTL_HOURS["head_to_head"])
    if h2h_data is None:
        h2h_data = asdict(
            provider.get_head_to_head_summary(
                fixture.home_team,
                fixture.away_team,
                fixture.home_team_id,
                fixture.away_team_id,
                home_snapshot,
                away_snapshot,
            )
        )
        cache.store("head_to_head", h2h_key, h2h_data)

    model = joblib.load(_model_path(model_name))
    X_live = _build_live_feature_row(fixture, home_data, away_data, h2h_data)
    prediction = model.predict(X_live)[0]

    response: dict[str, Any] = {
        "season": TARGET_SEASON,
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "home_snapshot_label": fixture.home_snapshot_label,
        "away_snapshot_label": fixture.away_snapshot_label,
        "fixture_utc_date": fixture.comparison_utc_date,
        "prediction": prediction,
        "features_used": X_live.to_dict(orient="records")[0],
        "data_summary": {
            "matchup": fixture_data,
            "home_recent_form": home_data,
            "away_recent_form": away_data,
            "head_to_head": h2h_data,
        },
    }

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(X_live)[0]
        response["probabilities"] = dict(zip(classes, probabilities))

    return response
