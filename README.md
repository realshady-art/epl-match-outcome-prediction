# Match Board: EPL Snapshot Outcome Prediction

Match Board is a Flask-based EPL prediction app for comparing two team snapshots. The model is trained on historical Premier League data, while each prediction assembles fresh pre-match context for a home-side snapshot and an away-side snapshot. Each side can use the latest available form with `now`, or a historical month such as `2025-01` or `2023-05`.

## Scope

- Competition: English Premier League
- Prediction surface: current or historical EPL team snapshots
- Training data: historical EPL data from prior seasons, plus current-season form for `now`
- Target: `H` / `D` / `A` for home win, draw, away win
- Constraint: use only information available before kickoff

## Current Feature Set

- Historical CSV training pipeline
- Match table cleaning and schema normalization
- Rolling pre-match feature engineering
- Season-aware train/validation/test split
- Baseline models
- Logistic Regression and Random Forest training
- Metric reporting and artifact export
- Snapshot-aware prediction service
- Local cache for reducing repeated data pulls
- Flask GUI with football-themed timeline and local archive
- Team dropdown selection instead of free-text club input
- Independent snapshot mode selectors for home and away sides

## Suggested Dataset

Use historical EPL CSV files from:

- `football-data.co.uk`: https://www.football-data.co.uk/data.php

Place season CSV files into:

```text
data/raw/
```

The pipeline expects columns that can be mapped to:

- `Date`
- `HomeTeam`
- `AwayTeam`
- `FTHG`
- `FTAG`
- `FTR`

These are standard columns in football-data.co.uk match files.

For the live 2025/26 prediction flow, the project will later call an external football API to pull:

- each team's last 5 matches
- recent head-to-head results
- rest days before the target fixture
- optional standings or form summaries

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── tables/
├── cache/
│   ├── head_to_head/
│   └── team_form/
├── src/
│   ├── __init__.py
│   ├── cache_manager.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── features.py
│   ├── live_data_provider.py
│   ├── predict_service.py
│   ├── train.py
│   └── utils.py
├── main.py
├── requirements.txt
└── epl_project_draft.txt
```

## Workflow

1. Add historical season CSVs to `data/raw/`
2. Run `python3 main.py train`
3. Review metrics under `outputs/tables/`
4. Review trained artifacts under `outputs/models/`
5. Use `python3 main.py predict --home TEAM --away TEAM`
6. Iterate on feature logic and API integration

## Feature Design

The training pipeline creates pre-match features such as:

- rolling points over last 3 and 5 matches
- rolling goals scored and conceded
- rolling goal difference
- rolling win rate
- rest days
- current rank proxy based on cumulative points before the match
- rank difference
- match month

The prediction service combines the trained model with fresh team context for the requested snapshots. The first live prediction version focuses on:

- last 5 match results for each team
- recent head-to-head results
- rest days before the comparison point

The project avoids leakage by computing features from information available before the current fixture.

## Modeling Approach

Baselines:

- most frequent class
- always home win

Machine learning models:

- multinomial logistic regression
- random forest classifier

## Validation Design

The historical training split is season-aware rather than randomly shuffled.

Default training split:

- train: `2015-2016` to `2022-2023`
- validation: `2023-2024`
- test: `2024-2025`

You can edit these values in `src/config.py`.

The current-season live source is configured as `2025-2026`, but user requests can also point at historical months.

## Output Files

After training, the project writes:

- `outputs/tables/model_metrics.csv`
- `outputs/tables/test_predictions.csv`
- `outputs/models/logistic_regression.joblib`
- `outputs/models/random_forest.joblib`

The prediction service will also use local cache files under `cache/` so repeated requests for the same teams do not call the API unnecessarily.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Live Data Setup

The live prediction service uses the current-season EPL CSV from `football-data.co.uk`, then merges it with historical CSVs to support arbitrary snapshot dates.

On the first prediction run, the provider downloads:

- `https://www.football-data.co.uk/mmz4281/2526/E0.csv`

It refreshes the file when the local copy is older than six hours.

The snapshot provider is intentionally limited to:

- each team's last 5 finished EPL matches prior to the requested snapshot
- rank and rest-day context from the snapshot season
- recent head-to-head results computed before the earlier of the two snapshot dates
- `now`, `YYYY-MM`, or `YYYY-MM-DD` snapshot inputs

## Run

```bash
python3 main.py fetch-data
python3 main.py train
python3 main.py predict --home Arsenal --away Chelsea
python3 main.py predict --home Arsenal --away "Man United" --home-date 2025-01 --away-date 2023-05
python3 main.py gui
```

If the same fixture context has already been fetched recently, the prediction flow reuses local cache files under `cache/` instead of repeating the same data pull logic.

## GUI

The local GUI runs on Flask with a SQLite backend.

Features included in V1:

- football-themed web form with a pitch-style matchup layout
- dropdown-based team selection for both sides
- independent `now` or `historical month` controls for each side
- automatic month-field enable/disable based on snapshot mode
- prediction result page with probability breakdown
- backend storage of each prediction request
- history page for previously stored predictions
- auto-refreshing timeline on the homepage

Start the app with:

```bash
python3 main.py gui
```

Then open:

```text
http://127.0.0.1:5000
```

## Next Steps

- wire in a real football API provider
- map live API responses into the existing feature schema
- add cache invalidation rules and TTLs
- add xG or shot-based features
- build a simple GUI around the prediction service

## Project Description

This project now targets a more flexible prediction surface than a single live fixture workflow. Instead of only asking for the next scheduled match, Match Board compares:

- `Team A` at `now` or a historical month
- `Team B` at `now` or a historical month

The prediction service then computes:

- recent form over the last 3 and 5 matches
- head-to-head context before the comparison cutoff
- rest-day and rank context at the selected snapshot

The Flask GUI is designed as a football product rather than a generic dashboard, with a pitch-driven input layout, a live prediction timeline, and a local archive of stored matchup snapshots.
