from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

from src.bootstrap_data import bootstrap_project_data
from src.cache_manager import CacheManager
from src.config import DATABASE_PATH, OUTPUT_MODELS_DIR
from src.live_data_provider import FootballDataCsvProvider, LiveDataProviderError
from src.predict_service import predict_match
from src.storage import (
    create_prediction_record,
    get_prediction_record,
    init_db,
    list_prediction_records,
)


def create_app(
    db_path: Path = DATABASE_PATH,
    cache_dir: Path | None = None,
    bootstrap_data: bool = True,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
    )

    if bootstrap_data:
        bootstrap_project_data()
    init_db(db_path)

    def _build_index_context(error: str | None = None, form_data: dict[str, str] | None = None) -> dict[str, Any]:
        provider = FootballDataCsvProvider()
        recent_predictions = list_prediction_records(limit=8, db_path=db_path)
        latest_result = recent_predictions[0] if recent_predictions else None
        return {
            "teams": provider.available_teams(),
            "recent_predictions": recent_predictions,
            "latest_result": latest_result,
            "model_ready": (OUTPUT_MODELS_DIR / "logistic_regression.joblib").exists(),
            "error": error,
            "form_data": form_data
            or {
                "home_team": "",
                "away_team": "",
                "home_snapshot_mode": "now",
                "away_snapshot_mode": "now",
                "home_snapshot_month": "",
                "away_snapshot_month": "",
            },
        }

    @app.get("/")
    def index() -> str:
        return render_template("index.html", **_build_index_context())

    @app.post("/predict")
    def predict_view():
        home_team = request.form.get("home_team", "").strip()
        away_team = request.form.get("away_team", "").strip()
        home_snapshot_mode = request.form.get("home_snapshot_mode", "now").strip()
        away_snapshot_mode = request.form.get("away_snapshot_mode", "now").strip()
        home_snapshot_month = request.form.get("home_snapshot_month", "").strip()
        away_snapshot_month = request.form.get("away_snapshot_month", "").strip()

        form_data = {
            "home_team": home_team,
            "away_team": away_team,
            "home_snapshot_mode": home_snapshot_mode,
            "away_snapshot_mode": away_snapshot_mode,
            "home_snapshot_month": home_snapshot_month,
            "away_snapshot_month": away_snapshot_month,
        }

        if not home_team or not away_team:
            return render_template("index.html", **_build_index_context(error="Enter both team names before requesting a prediction.", form_data=form_data))

        if home_team == away_team:
            return render_template("index.html", **_build_index_context(error="Choose two different clubs for the comparison.", form_data=form_data))

        if home_snapshot_mode == "historical" and not home_snapshot_month:
            return render_template("index.html", **_build_index_context(error="Select a month for the home team snapshot, or switch it to Now.", form_data=form_data))

        if away_snapshot_mode == "historical" and not away_snapshot_month:
            return render_template("index.html", **_build_index_context(error="Select a month for the away team snapshot, or switch it to Now.", form_data=form_data))

        home_snapshot = home_snapshot_month if home_snapshot_mode == "historical" else "now"
        away_snapshot = away_snapshot_month if away_snapshot_mode == "historical" else "now"

        try:
            result = predict_match(
                home_team=home_team,
                away_team=away_team,
                home_snapshot=home_snapshot,
                away_snapshot=away_snapshot,
                provider=FootballDataCsvProvider(),
                cache=CacheManager(cache_dir) if cache_dir is not None else None,
            )
        except FileNotFoundError:
            return render_template("index.html", **_build_index_context(error="No trained model found. Run the training pipeline first.", form_data=form_data))
        except LiveDataProviderError as exc:
            return render_template("index.html", **_build_index_context(error=str(exc), form_data=form_data))
        except Exception as exc:
            return render_template(
                "index.html",
                **_build_index_context(
                    error=f"Prediction request failed: {exc}",
                    form_data=form_data,
                ),
            )

        record_id = create_prediction_record(
            created_at=datetime.now(timezone.utc).isoformat(),
            home_team=result["home_team"],
            away_team=result["away_team"],
            home_snapshot_label=result["home_snapshot_label"],
            away_snapshot_label=result["away_snapshot_label"],
            fixture_utc_date=result["fixture_utc_date"],
            prediction=result["prediction"],
            probabilities=result.get("probabilities"),
            features=result["features_used"],
            summary=result["data_summary"],
            db_path=db_path,
        )
        return redirect(url_for("prediction_detail", record_id=record_id))

    @app.get("/history")
    def history() -> str:
        records = list_prediction_records(limit=100, db_path=db_path)
        return render_template(
            "history.html",
            records=records,
            record_count=len(records),
            model_ready=(OUTPUT_MODELS_DIR / "logistic_regression.joblib").exists(),
        )

    @app.get("/predictions/<int:record_id>")
    def prediction_detail(record_id: int) -> str:
        record = get_prediction_record(record_id, db_path=db_path)
        if record is None:
            abort(404)
        return render_template(
            "detail.html",
            record=record,
            model_ready=(OUTPUT_MODELS_DIR / "logistic_regression.joblib").exists(),
        )

    @app.get("/api/timeline")
    def timeline_api():
        records = list_prediction_records(limit=12, db_path=db_path)
        return jsonify(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": records,
            }
        )

    return app
