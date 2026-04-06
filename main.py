import argparse
import json
from pathlib import Path

from src.bootstrap_data import bootstrap_project_data
from src.config import APP_HOST, APP_PORT, OUTPUT_TABLES_DIR
from src.gui_app import create_app
from src.predict_service import predict_match
from src.train import run_training_pipeline


def run_train() -> None:
    metrics = run_training_pipeline()
    metrics_path = Path(OUTPUT_TABLES_DIR) / "model_metrics.csv"
    print("Training complete.")
    print(f"Metrics saved to: {metrics_path}")
    print(metrics.to_string(index=False))


def run_predict(home_team: str, away_team: str, home_snapshot: str, away_snapshot: str) -> None:
    prediction = predict_match(
        home_team=home_team,
        away_team=away_team,
        home_snapshot=home_snapshot,
        away_snapshot=away_snapshot,
    )
    print(json.dumps(prediction, indent=2))


def run_fetch_data() -> None:
    downloaded = bootstrap_project_data()
    if not downloaded:
        print("All required data files are already present.")
        return
    print("Downloaded files:")
    for path in downloaded:
        print(path)


def run_gui(host: str, port: int) -> None:
    app = create_app()
    app.run(host=host, port=port, debug=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EPL 2025/26 match prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch-data", help="Download historical and live EPL CSV data")
    subparsers.add_parser("train", help="Train historical EPL models")
    gui_parser = subparsers.add_parser("gui", help="Run the local Match Board web interface")
    gui_parser.add_argument("--host", default=APP_HOST, help="Host for the local web server")
    gui_parser.add_argument("--port", default=APP_PORT, type=int, help="Port for the local web server")

    predict_parser = subparsers.add_parser("predict", help="Predict from two EPL team snapshots")
    predict_parser.add_argument("--home", required=True, dest="home_team", help="Home team name")
    predict_parser.add_argument("--away", required=True, dest="away_team", help="Away team name")
    predict_parser.add_argument(
        "--home-date",
        default="now",
        dest="home_snapshot",
        help="Snapshot for the home team: now, YYYY-MM, or YYYY-MM-DD",
    )
    predict_parser.add_argument(
        "--away-date",
        default="now",
        dest="away_snapshot",
        help="Snapshot for the away team: now, YYYY-MM, or YYYY-MM-DD",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch-data":
        run_fetch_data()
    elif args.command == "train":
        run_train()
    elif args.command == "gui":
        run_gui(host=args.host, port=args.port)
    elif args.command == "predict":
        run_predict(
            home_team=args.home_team,
            away_team=args.away_team,
            home_snapshot=args.home_snapshot,
            away_snapshot=args.away_snapshot,
        )


if __name__ == "__main__":
    main()
