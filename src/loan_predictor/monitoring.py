"""Evidently drift + data-quality monitoring for the loan-approval model.

Compares a CURRENT dataset against a REFERENCE (the data the model trained on)
and produces:
  - a visual HTML report (per-feature distributions, drift scores),
  - a machine-readable JSON summary,
  - a PASS/FAIL verdict from Evidently's built-in test suite.

This answers: "Is the incoming data still safe for the current model?" — the
signal that tells us WHEN to retrain. Uses Evidently 0.7+ (Dataset/Report API).

Demo:
    python -m loan_predictor.monitoring
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
CLEAN = DATA_DIR / "processed/hmda_clean.parquet"
REPORTS_DIR = ROOT / "reports"
HMDA_URL = ("https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
            "?states={states}&years={year}&actions_taken=1,3")

# The features the model uses (must match training; reference & current share this schema).
NUM = ["loan_amount", "loan_to_value_ratio", "property_value", "income",
       "loan_term", "dti", "loan_to_income"]
CAT = ["loan_type", "loan_purpose", "lien_status", "occupancy_type"]
FEATURES = NUM + CAT

# The core risk drivers — drift in ANY of these alone is enough to flag a retrain.
CRITICAL_FEATURES = {"income", "dti", "loan_to_value_ratio"}
# Or flag if more than this share of all columns drift (env-tunable).
DRIFT_SHARE_THRESHOLD = float(os.environ.get("DRIFT_SHARE_THRESHOLD", "0.3"))


def _schema() -> DataDefinition:
    return DataDefinition(numerical_columns=NUM, categorical_columns=CAT)


def _status(test: dict) -> str:
    s = test["status"]
    return getattr(s, "value", str(s)).upper()


WORKSPACE_DIR = Path(os.environ.get("EVIDENTLY_WORKSPACE", ROOT / "evidently_workspace"))


def publish_to_workspace(result, name: str = "loan-drift-check"):
    """Add a run to a LOCAL Evidently workspace (files on disk) so it shows up
    as graphs over time in the self-hosted Evidently UI.

    (Evidently Cloud was shut down in 2026 -> we self-host the free UI instead.)
    View with:  evidently ui --workspace evidently_workspace   -> http://127.0.0.1:8000
    Runs accumulate in one project so you get trends; the project id is cached
    in the workspace so repeated runs land in the same project."""
    from evidently.ui.workspace import Workspace
    ws = Workspace.create(str(WORKSPACE_DIR))
    pid_file = WORKSPACE_DIR / "project_id.txt"
    project_id = (os.environ.get("EVIDENTLY_PROJECT_ID")
                  or (pid_file.read_text().strip() if pid_file.exists() else None))
    if not project_id:
        project = ws.create_project(os.environ.get("EVIDENTLY_PROJECT_NAME", "Loan Approval Monitoring"))
        project_id = str(project.id)
        pid_file.write_text(project_id)
        print(f"  Created Evidently project {project_id}")
    ws.add_run(project_id, result, include_data=False, name=name)
    print(f"  Added run '{name}' to workspace '{WORKSPACE_DIR}' (project {project_id})")
    return project_id


def run_monitoring(reference: pd.DataFrame, current: pd.DataFrame,
                   html_path: str | None = None, json_path: str | None = None,
                   publish_name: str | None = None) -> dict:
    """Compare current vs reference. Returns a summary dict with the PASS/FAIL verdict.

    If publish_name is given, the run is also added to the local Evidently
    workspace (viewable as graphs-over-time in the self-hosted Evidently UI)."""
    schema = _schema()
    ref = Dataset.from_pandas(reference[FEATURES], data_definition=schema)
    cur = Dataset.from_pandas(current[FEATURES], data_definition=schema)

    # Drift (distribution shift) + data summary/quality, with built-in pass/fail tests.
    report = Report([DataDriftPreset(), DataSummaryPreset()], include_tests=True)
    result = report.run(cur, ref)

    if html_path:
        result.save_html(str(html_path))
    if publish_name:
        publish_to_workspace(result, publish_name)

    tests = result.dict()["tests"]
    # Gate the verdict on DRIFT only — the real "should we retrain?" signal.
    # (DataSummaryPreset's exact-stat tests are too strict for monitoring; they
    #  stay in the HTML as visual context, not as pass/fail gates.)
    value_drift = [t for t in tests if t["name"].startswith("Value Drift for column")]
    drifted_columns = [t["name"].replace("Value Drift for column ", "")
                       for t in value_drift if _status(t) == "FAIL"]
    share = len(drifted_columns) / len(value_drift) if value_drift else 0.0
    critical_drift = sorted(c for c in drifted_columns if c in CRITICAL_FEATURES)
    # Retrain signal: too many columns drifted, OR any core risk feature drifted.
    passed = (share <= DRIFT_SHARE_THRESHOLD) and (not critical_drift)

    summary = {
        "passed": passed,
        "drift_share": round(share, 3),
        "share_threshold": DRIFT_SHARE_THRESHOLD,
        "n_drifted_columns": len(drifted_columns),
        "total_columns": len(value_drift),
        "drifted_columns": drifted_columns,
        "critical_drift": critical_drift,
        "reference_rows": len(reference),
        "current_rows": len(current),
    }
    if json_path:
        Path(json_path).write_text(json.dumps(summary, indent=2))
    return summary


def _print(label: str, summary: dict) -> None:
    verdict = "PASS ✅ (data is safe)" if summary["passed"] else "FAIL ❌ (drift — retrain)"
    print(f"\n=== {label}: {verdict} ===")
    print(f"  drift share: {summary['drift_share']} (threshold {summary['share_threshold']})")
    print(f"  drifted columns: {summary['n_drifted_columns']}/{summary['total_columns']}"
          f" -> {summary['drifted_columns']}")
    print(f"  CRITICAL features drifted: {summary['critical_drift'] or 'none'}")


def _fetch_clean(states: str, year: str = "2023", sample: int = 15000) -> pd.DataFrame:
    """Download a state slice of HMDA, clean it (same pipeline as training),
    and return a sample. Caches the raw file so repeat runs don't re-download."""
    from loan_predictor.clean import clean
    tag = states.replace(",", "_")
    raw = DATA_DIR / "raw" / f"hmda_{tag}_{year}.csv"
    out = DATA_DIR / "processed" / f"clean_{tag}_{year}.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print(f"  downloading HMDA {states} {year}...")
        urllib.request.urlretrieve(HMDA_URL.format(states=states, year=year), raw)
    clean(str(raw), str(out))
    df = pd.read_parquet(out)
    return df.sample(sample, random_state=42) if len(df) > sample else df


PAGES_URL = "https://nickychemos.github.io/Loan-Approval-Predictor/"


def notify_slack(summary: dict) -> None:
    """Post a Slack alert ONLY when drift is detected (no spam on healthy runs).
    No-op unless SLACK_WEBHOOK_URL is set."""
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url or summary.get("passed"):
        return
    drifted = summary.get("critical_drift") or summary.get("drifted_columns") or []
    text = (":warning: *Loan model — data drift detected*\n"
            f"Drift share: {summary['drift_share']} (threshold {summary['share_threshold']})\n"
            f"Drifted: {', '.join(drifted)}\n"
            f"Consider retraining. Report: {PAGES_URL}")
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
        print("  Slack alert sent.")
    except Exception as e:
        print(f"  Slack alert failed: {e}")


def monitoring_job() -> dict:
    """Production monitoring run: compare a CURRENT data slice against the
    training-region REFERENCE, write the report + verdict, and publish locally.
    In production, point CURRENT at live Postgres applications (env-configurable)."""
    ref_states = os.environ.get("MONITOR_REFERENCE_STATES", "MD")
    cur_states = os.environ.get("MONITOR_CURRENT_STATES", "DC")
    print(f"Monitoring: reference={ref_states}  current={cur_states}")
    reference = _fetch_clean(ref_states)
    current = _fetch_clean(cur_states)
    REPORTS_DIR.mkdir(exist_ok=True)
    summary = run_monitoring(reference, current,
                             html_path=REPORTS_DIR / "monitoring.html",
                             json_path=REPORTS_DIR / "monitoring.json",
                             publish_name=f"monitor-{cur_states}")
    _print(f"Monitoring (ref={ref_states} vs current={cur_states})", summary)
    notify_slack(summary)
    return summary


if __name__ == "__main__":
    if "job" in sys.argv:               # production-style run (used by CI)
        monitoring_job()
        sys.exit(0)
    # otherwise: the local two-scenario teaching demo
    REPORTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(CLEAN)
    reference = df.sample(8000, random_state=1)

    # Scenario A — a fresh slice of the SAME population: expect NO drift -> PASS.
    current_ok = df.sample(8000, random_state=2)
    s_ok = run_monitoring(reference, current_ok,
                          html_path=REPORTS_DIR / "monitoring_nodrift.html",
                          publish_name="no-drift-check")
    _print("Scenario A (similar data)", s_ok)

    # Scenario B — a DRIFTED population (e.g. a downturn: lower incomes, higher debt
    # & loan-to-value): expect drift detected -> FAIL.
    current_drift = df.sample(8000, random_state=3).copy()
    current_drift["income"] = current_drift["income"] * 0.5
    current_drift["dti"] = current_drift["dti"] + 25
    current_drift["loan_to_value_ratio"] = current_drift["loan_to_value_ratio"] + 20
    s_drift = run_monitoring(reference, current_drift,
                             html_path=REPORTS_DIR / "monitoring_drift.html",
                             json_path=REPORTS_DIR / "monitoring.json",
                             publish_name="drift-check")
    _print("Scenario B (drifted data)", s_drift)
    print(f"\nHTML reports saved in {REPORTS_DIR}/ (open monitoring_drift.html to see it).")
