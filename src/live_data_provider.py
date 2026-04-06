from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from src.config import (
    DATE_COLUMN,
    HOME_TEAM_COLUMN,
    LIVE_DATA_DIR,
    LIVE_DATA_FILENAME,
    LIVE_DATA_URL,
    SEASON_COLUMN,
    TARGET_COLUMN,
    TARGET_SEASON,
    AWAY_TEAM_COLUMN,
)
from src.data_loader import load_all_raw_data
from src.features import _points_from_result
from src.utils import ensure_directories


@dataclass
class MatchupContext:
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_snapshot_label: str
    away_snapshot_label: str
    comparison_utc_date: str
    season: str
    home_rank: int
    away_rank: int
    comparison_month: int


@dataclass
class TeamRecentForm:
    team: str
    team_id: int
    snapshot_label: str
    season: str
    wins_last_3: int
    draws_last_3: int
    losses_last_3: int
    points_last_3: int
    goals_for_last_3: int
    goals_against_last_3: int
    wins_last_5: int
    draws_last_5: int
    losses_last_5: int
    points_last_5: int
    goals_for_last_5: int
    goals_against_last_5: int
    rest_days: int
    rank: int
    last_match_utc_date: str


@dataclass
class HeadToHeadSummary:
    home_team: str
    away_team: str
    cutoff_label: str
    home_points_last_3: int
    away_points_last_3: int
    home_goal_diff_last_3: int
    sample_size: int


class LiveDataProvider(Protocol):
    def available_teams(self) -> list[str]:
        ...

    def get_matchup_context(
        self,
        home_team: str,
        away_team: str,
        home_snapshot: str = "now",
        away_snapshot: str = "now",
    ) -> MatchupContext:
        ...

    def get_team_recent_form(self, team_name: str, team_id: int, snapshot_value: str) -> TeamRecentForm:
        ...

    def get_head_to_head_summary(
        self,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        home_snapshot: str,
        away_snapshot: str,
    ) -> HeadToHeadSummary:
        ...


class LiveDataProviderError(RuntimeError):
    pass


class FootballDataCsvProvider:
    """Snapshot-based provider backed by historical EPL CSVs plus the current-season CSV."""

    TEAM_ALIASES = {
        "man city": "Man City",
        "manchester city": "Man City",
        "man utd": "Man United",
        "manchester united": "Man United",
        "newcastle": "Newcastle",
        "newcastle united": "Newcastle",
        "spurs": "Tottenham",
        "tottenham hotspur": "Tottenham",
        "wolves": "Wolves",
        "wolverhampton": "Wolves",
        "wolverhampton wanderers": "Wolves",
        "nottingham forest": "Nott'm Forest",
        "forest": "Nott'm Forest",
        "sheffield united": "Sheffield United",
        "ipswich town": "Ipswich",
        "leicester city": "Leicester",
        "west brom": "West Brom",
        "brighton": "Brighton",
        "brighton & hove albion": "Brighton",
    }

    def __init__(self, live_csv_path: Path | None = None) -> None:
        self.live_csv_path = live_csv_path or (LIVE_DATA_DIR / LIVE_DATA_FILENAME)
        self._live_df: pd.DataFrame | None = None
        self._historical_df: pd.DataFrame | None = None
        self._all_matches_df: pd.DataFrame | None = None
        self._team_ids: dict[str, int] = {}

    def available_teams(self) -> list[str]:
        frame = self._get_all_matches_df()
        return sorted(set(frame[HOME_TEAM_COLUMN]).union(set(frame[AWAY_TEAM_COLUMN])))

    def get_matchup_context(
        self,
        home_team: str,
        away_team: str,
        home_snapshot: str = "now",
        away_snapshot: str = "now",
    ) -> MatchupContext:
        resolved_home = self._resolve_team_name(home_team)
        resolved_away = self._resolve_team_name(away_team)
        home_label, home_cutoff = self._parse_snapshot_input(home_snapshot)
        away_label, away_cutoff = self._parse_snapshot_input(away_snapshot)

        home_form = self.get_team_recent_form(resolved_home, self._team_id(resolved_home), home_snapshot)
        away_form = self.get_team_recent_form(resolved_away, self._team_id(resolved_away), away_snapshot)

        comparison_cutoff = max(home_cutoff, away_cutoff)
        comparison_display = self._to_utc_date_string(pd.Timestamp(comparison_cutoff - timedelta(days=1)))

        return MatchupContext(
            home_team=resolved_home,
            away_team=resolved_away,
            home_team_id=self._team_id(resolved_home),
            away_team_id=self._team_id(resolved_away),
            home_snapshot_label=home_label,
            away_snapshot_label=away_label,
            comparison_utc_date=comparison_display,
            season=TARGET_SEASON,
            home_rank=home_form.rank,
            away_rank=away_form.rank,
            comparison_month=(comparison_cutoff - timedelta(days=1)).month,
        )

    def get_team_recent_form(self, team_name: str, team_id: int, snapshot_value: str) -> TeamRecentForm:
        del team_id
        snapshot_label, cutoff = self._parse_snapshot_input(snapshot_value)
        history = self._team_matches_before_cutoff(self._get_all_matches_df(), team_name, cutoff)
        if history.empty:
            raise LiveDataProviderError(f"No completed matches found for {team_name} before {snapshot_label}.")

        active_season = str(history.iloc[0][SEASON_COLUMN])
        season_history = history.loc[history[SEASON_COLUMN] == active_season]
        last_5 = season_history.head(5)
        last_3 = season_history.head(3)

        summary_5 = self._summarize_team_matches(last_5, team_name)
        summary_3 = self._summarize_team_matches(last_3, team_name)

        last_match_date = last_5.iloc[0][DATE_COLUMN]
        rest_days = max((cutoff.date() - last_match_date.date()).days, 0)
        standings = self._compute_standings(active_season, cutoff)

        return TeamRecentForm(
            team=team_name,
            team_id=self._team_id(team_name),
            snapshot_label=snapshot_label,
            season=active_season,
            wins_last_3=summary_3["wins"],
            draws_last_3=summary_3["draws"],
            losses_last_3=summary_3["losses"],
            points_last_3=summary_3["points"],
            goals_for_last_3=summary_3["goals_for"],
            goals_against_last_3=summary_3["goals_against"],
            wins_last_5=summary_5["wins"],
            draws_last_5=summary_5["draws"],
            losses_last_5=summary_5["losses"],
            points_last_5=summary_5["points"],
            goals_for_last_5=summary_5["goals_for"],
            goals_against_last_5=summary_5["goals_against"],
            rest_days=rest_days,
            rank=standings.get(team_name, 0),
            last_match_utc_date=self._to_utc_date_string(last_match_date),
        )

    def get_head_to_head_summary(
        self,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        home_snapshot: str,
        away_snapshot: str,
    ) -> HeadToHeadSummary:
        del home_team_id, away_team_id
        home_label, home_cutoff = self._parse_snapshot_input(home_snapshot)
        away_label, away_cutoff = self._parse_snapshot_input(away_snapshot)
        comparison_cutoff = min(home_cutoff, away_cutoff)
        comparison_label = home_label if home_cutoff <= away_cutoff else away_label

        historical = self._get_all_matches_df()
        mask = (
            (
                (historical[HOME_TEAM_COLUMN] == home_team)
                & (historical[AWAY_TEAM_COLUMN] == away_team)
            )
            | (
                (historical[HOME_TEAM_COLUMN] == away_team)
                & (historical[AWAY_TEAM_COLUMN] == home_team)
            )
        ) & (historical[DATE_COLUMN] < comparison_cutoff)

        recent = historical.loc[mask].sort_values(DATE_COLUMN, ascending=False).head(3)

        home_points = 0
        away_points = 0
        home_goal_diff = 0
        for _, row in recent.iterrows():
            if row[HOME_TEAM_COLUMN] == home_team:
                home_side = "home"
                away_side = "away"
                home_goals = row["home_goals"]
                away_goals = row["away_goals"]
            else:
                home_side = "away"
                away_side = "home"
                home_goals = row["away_goals"]
                away_goals = row["home_goals"]

            home_points += _points_from_result(row[TARGET_COLUMN], home_side)
            away_points += _points_from_result(row[TARGET_COLUMN], away_side)
            home_goal_diff += home_goals - away_goals

        return HeadToHeadSummary(
            home_team=home_team,
            away_team=away_team,
            cutoff_label=comparison_label,
            home_points_last_3=home_points,
            away_points_last_3=away_points,
            home_goal_diff_last_3=home_goal_diff,
            sample_size=len(recent),
        )

    def _get_live_df(self) -> pd.DataFrame:
        if self._live_df is None:
            self._refresh_live_csv_if_needed()
            df = pd.read_csv(self.live_csv_path)
            df = self._normalize_match_frame(df)
            df[SEASON_COLUMN] = TARGET_SEASON
            self._live_df = df
        return self._live_df

    def _get_historical_df(self) -> pd.DataFrame:
        if self._historical_df is None:
            self._historical_df = load_all_raw_data()
        return self._historical_df

    def _get_all_matches_df(self) -> pd.DataFrame:
        if self._all_matches_df is None:
            live_finished = self._get_live_df().loc[self._get_live_df()[TARGET_COLUMN].notna()].copy()
            historical = self._get_historical_df()
            self._all_matches_df = (
                pd.concat([historical, live_finished], ignore_index=True)
                .sort_values([DATE_COLUMN, HOME_TEAM_COLUMN, AWAY_TEAM_COLUMN])
                .reset_index(drop=True)
            )
        return self._all_matches_df

    def _refresh_live_csv_if_needed(self) -> None:
        ensure_directories([LIVE_DATA_DIR])
        refresh = True
        if self.live_csv_path.exists():
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                self.live_csv_path.stat().st_mtime,
                tz=timezone.utc,
            )
            refresh = age > timedelta(hours=6)

        if refresh:
            try:
                response = requests.get(LIVE_DATA_URL, timeout=20)
                response.raise_for_status()
            except requests.RequestException:
                if self.live_csv_path.exists():
                    return
                raise LiveDataProviderError(
                    "Could not refresh the current-season CSV and no local live file is available."
                )
            self.live_csv_path.write_text(response.text, encoding="utf-8")

    def _compute_standings(self, season: str, cutoff: datetime) -> dict[str, int]:
        completed = self._get_all_matches_df()
        completed = completed[
            (completed[SEASON_COLUMN] == season)
            & completed[TARGET_COLUMN].notna()
            & (completed[DATE_COLUMN] < cutoff)
        ].sort_values(DATE_COLUMN)

        table: dict[str, dict[str, int]] = {}
        for _, row in completed.iterrows():
            home = row[HOME_TEAM_COLUMN]
            away = row[AWAY_TEAM_COLUMN]
            table.setdefault(home, {"points": 0, "goal_diff": 0, "goals_for": 0})
            table.setdefault(away, {"points": 0, "goal_diff": 0, "goals_for": 0})

            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            table[home]["points"] += _points_from_result(row[TARGET_COLUMN], "home")
            table[away]["points"] += _points_from_result(row[TARGET_COLUMN], "away")
            table[home]["goal_diff"] += home_goals - away_goals
            table[away]["goal_diff"] += away_goals - home_goals
            table[home]["goals_for"] += home_goals
            table[away]["goals_for"] += away_goals

        ordered = sorted(
            table.items(),
            key=lambda item: (item[1]["points"], item[1]["goal_diff"], item[1]["goals_for"]),
            reverse=True,
        )
        return {team: idx + 1 for idx, (team, _) in enumerate(ordered)}

    def _team_matches_before_cutoff(self, matches: pd.DataFrame, team_name: str, cutoff: datetime) -> pd.DataFrame:
        mask = (
            matches[TARGET_COLUMN].notna()
            & (matches[DATE_COLUMN] < cutoff)
            & ((matches[HOME_TEAM_COLUMN] == team_name) | (matches[AWAY_TEAM_COLUMN] == team_name))
        )
        return matches.loc[mask].sort_values(DATE_COLUMN, ascending=False)

    def _summarize_team_matches(self, matches: pd.DataFrame, team_name: str) -> dict[str, int]:
        wins = draws = losses = goals_for = goals_against = 0
        for _, row in matches.iterrows():
            is_home = row[HOME_TEAM_COLUMN] == team_name
            team_goals = int(row["home_goals"] if is_home else row["away_goals"])
            opponent_goals = int(row["away_goals"] if is_home else row["home_goals"])
            goals_for += team_goals
            goals_against += opponent_goals
            if team_goals > opponent_goals:
                wins += 1
            elif team_goals < opponent_goals:
                losses += 1
            else:
                draws += 1

        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "points": wins * 3 + draws,
            "goals_for": goals_for,
            "goals_against": goals_against,
        }

    def _team_id(self, team_name: str) -> int:
        if not self._team_ids:
            teams = self.available_teams()
            self._team_ids = {team: idx + 1 for idx, team in enumerate(teams)}
        return self._team_ids[team_name]

    def _resolve_team_name(self, team_name: str) -> str:
        teams = self.available_teams()
        normalized_map = {team.casefold(): team for team in teams}
        alias = self.TEAM_ALIASES.get(team_name.casefold(), team_name)
        if alias.casefold() in normalized_map:
            return normalized_map[alias.casefold()]

        partial = [team for team in teams if alias.casefold() in team.casefold() or team.casefold() in alias.casefold()]
        if len(partial) == 1:
            return partial[0]

        raise LiveDataProviderError(f"Could not resolve team name from available data: {team_name}")

    def _parse_snapshot_input(self, snapshot_value: str | None) -> tuple[str, datetime]:
        if snapshot_value is None or not str(snapshot_value).strip() or str(snapshot_value).strip().casefold() == "now":
            latest_completed = self._get_all_matches_df()[DATE_COLUMN].max()
            if pd.isna(latest_completed):
                raise LiveDataProviderError("No completed matches available for snapshot resolution.")
            cutoff = pd.Timestamp(latest_completed).to_pydatetime() + timedelta(days=1)
            return "Now", cutoff

        value = str(snapshot_value).strip()
        if len(value) == 7:
            try:
                period = pd.Period(value, freq="M")
            except ValueError as exc:
                raise LiveDataProviderError(f"Invalid snapshot month: {snapshot_value}") from exc
            cutoff = period.to_timestamp(how="end").to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            return period.strftime("%b %Y"), cutoff

        try:
            cutoff_date = datetime.fromisoformat(value)
        except ValueError as exc:
            raise LiveDataProviderError(
                f"Invalid snapshot value '{snapshot_value}'. Use 'now', YYYY-MM, or YYYY-MM-DD."
            ) from exc
        cutoff = cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return cutoff_date.strftime("%d %b %Y"), cutoff

    @staticmethod
    def _normalize_match_frame(df: pd.DataFrame) -> pd.DataFrame:
        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise LiveDataProviderError(f"Live CSV missing required columns: {missing}")

        normalized = df[required].rename(
            columns={
                "Date": DATE_COLUMN,
                "HomeTeam": HOME_TEAM_COLUMN,
                "AwayTeam": AWAY_TEAM_COLUMN,
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": TARGET_COLUMN,
            }
        )
        normalized[DATE_COLUMN] = pd.to_datetime(normalized[DATE_COLUMN], dayfirst=True, errors="coerce")
        normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].astype("string").str.strip().str.upper()
        return normalized.dropna(subset=[DATE_COLUMN, HOME_TEAM_COLUMN, AWAY_TEAM_COLUMN]).reset_index(drop=True)

    @staticmethod
    def _to_utc_date_string(value: pd.Timestamp | datetime) -> str:
        dt = pd.Timestamp(value).to_pydatetime().replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
