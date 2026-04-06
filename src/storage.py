from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config import DATABASE_PATH
from src.utils import ensure_directories


def _connect(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    ensure_directories([db_path.parent])
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path = DATABASE_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_snapshot_label TEXT,
                away_snapshot_label TEXT,
                fixture_utc_date TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probabilities_json TEXT,
                features_json TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "home_snapshot_label" not in existing_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN home_snapshot_label TEXT")
        if "away_snapshot_label" not in existing_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN away_snapshot_label TEXT")
        connection.commit()


def create_prediction_record(
    *,
    created_at: str,
    home_team: str,
    away_team: str,
    home_snapshot_label: str,
    away_snapshot_label: str,
    fixture_utc_date: str,
    prediction: str,
    probabilities: dict[str, float] | None,
    features: dict[str, Any],
    summary: dict[str, Any],
    db_path: Path = DATABASE_PATH,
) -> int:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                created_at,
                home_team,
                away_team,
                home_snapshot_label,
                away_snapshot_label,
                fixture_utc_date,
                prediction,
                probabilities_json,
                features_json,
                summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                home_team,
                away_team,
                home_snapshot_label,
                away_snapshot_label,
                fixture_utc_date,
                prediction,
                json.dumps(probabilities or {}),
                json.dumps(features),
                json.dumps(summary),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def list_prediction_records(limit: int = 50, db_path: Path = DATABASE_PATH) -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, home_team, away_team, home_snapshot_label, away_snapshot_label, fixture_utc_date, prediction, probabilities_json
            FROM predictions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    records = []
    for row in rows:
        records.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_snapshot_label": row["home_snapshot_label"] or "Now",
                "away_snapshot_label": row["away_snapshot_label"] or "Now",
                "fixture_utc_date": row["fixture_utc_date"],
                "prediction": row["prediction"],
                "probabilities": json.loads(row["probabilities_json"] or "{}"),
            }
        )
    return records


def get_prediction_record(record_id: int, db_path: Path = DATABASE_PATH) -> dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM predictions
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_snapshot_label": row["home_snapshot_label"] or "Now",
        "away_snapshot_label": row["away_snapshot_label"] or "Now",
        "fixture_utc_date": row["fixture_utc_date"],
        "prediction": row["prediction"],
        "probabilities": json.loads(row["probabilities_json"] or "{}"),
        "features": json.loads(row["features_json"]),
        "summary": json.loads(row["summary_json"]),
    }
