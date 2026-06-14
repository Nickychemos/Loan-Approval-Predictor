"""Prefect orchestration for the loan-approval ML pipeline.

VOCABULARY
  - @task : one step in the pipeline (clean, train, validate, promote). Prefect
            tracks its state (Pending -> Running -> Completed/Failed), retries it
            on failure, and logs everything.
  - @flow : the pipeline itself. It calls tasks; Prefect works out the order from
            how their results feed each other, and gives you one run record.

THE RETRAIN FLOW (what each run does)
  1. clean_data      -> rebuild the tidy parquet from raw HMDA data
  2. tune_model      -> hyperparameter-search + train a CANDIDATE model
  3. validate_model  -> gate: is the candidate at least as good as production?
  4. promote_model   -> only if it passes, copy candidate -> production

Run once:        python -m loan_predictor.flows
Serve on a cron: python -m loan_predictor.flows serve   (Sundays 02:00)
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Make the ML package importable no matter where/how this script is launched
# (locally, or pulled fresh onto a cloud worker by Prefect Cloud).
sys.path.insert(0, str(ROOT / "src"))

from prefect import flow, task, get_run_logger

# Where the loan data comes from when it isn't already on disk (CFPB HMDA API).
HMDA_URL = ("https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
            "?states=MD,VA,CO,OR,TN,MO&years=2023&actions_taken=1,3")
# Folders are env-overridable so a cloud/VM run can point at a PERSISTENT disk
# (e.g. MODELS_DIR=/data/models on the Oracle VM) without changing code.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
REPORTS_DIR = ROOT / "reports"
RAW = DATA_DIR / "raw/hmda_multistate_2023.csv"
CLEAN = DATA_DIR / "processed/hmda_clean.parquet"
CANDIDATE = MODELS_DIR / "loan_approval_model_candidate.joblib"
PRODUCTION = MODELS_DIR / "loan_approval_model_tuned.joblib"
BEST_PARAMS = REPORTS_DIR / "best_params.json"
PROD_META = REPORTS_DIR / "production_meta.json"
MIN_PR_AUC = 0.90   # absolute floor; the real gate is "beat current production"


@task(retries=3, retry_delay_seconds=30)
def download_data() -> str:
    """Step 0 — make sure the raw data exists. Downloads it from the CFPB HMDA
    API if missing, so a fresh cloud machine can fetch the data by itself.
    Skips if already on disk (idempotent — only the first run pays the cost)."""
    logger = get_run_logger()
    if RAW.exists():
        logger.info(f"Raw data already present ({RAW.name}) — skipping download.")
        return str(RAW)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HMDA data from the CFPB API (first run only)...")
    urllib.request.urlretrieve(HMDA_URL, RAW)
    logger.info(f"Downloaded {RAW.stat().st_size / 1e6:.0f} MB to {RAW.name}")
    return str(RAW)


@task(retries=2, retry_delay_seconds=15)
def clean_data(raw_path: str) -> str:
    """Rebuild the model-ready dataset (dedup, feature-engineer, winsorise)."""
    logger = get_run_logger()
    logger.info("Cleaning raw HMDA data...")
    from loan_predictor.clean import clean
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    clean(raw_path, str(CLEAN))
    logger.info(f"Clean data written to {CLEAN.name}")
    return str(CLEAN)


@task(retries=1, retry_delay_seconds=30)
def tune_model(data_path: str, sample: int, n_iter: int) -> str:
    """Step 2 — search hyperparameters and train a CANDIDATE model (not production)."""
    logger = get_run_logger()
    logger.info(f"Tuning candidate model (search sample={sample:,}, n_iter={n_iter})...")
    from loan_predictor.tune import main as tune_main
    tune_main(data_path, sample=sample, n_iter=n_iter, out_model=str(CANDIDATE))
    logger.info(f"Candidate model written to {CANDIDATE.name}")
    return str(CANDIDATE)


@task
def validate_model(candidate_path: str) -> dict:
    """Step 3 — the GATE. Promote only if the candidate beats the current
    production model (and clears an absolute floor)."""
    logger = get_run_logger()
    bp = json.loads(BEST_PARAMS.read_text())
    candidate_pr = bp["test_pr_auc"]
    prod_pr = 0.0
    if PROD_META.exists():
        prod_pr = json.loads(PROD_META.read_text()).get("test_pr_auc", 0.0)
    gate = max(MIN_PR_AUC, prod_pr)
    passed = candidate_pr >= gate
    logger.info(f"Candidate PR-AUC={candidate_pr}  |  gate(max floor/prod)={gate}  "
                f"=> {'PASS' if passed else 'REJECT'}")
    return {"candidate_path": candidate_path, "candidate_pr": candidate_pr,
            "gate": gate, "passed": passed,
            "candidate_params": bp.get("best_params", {}),
            "n_iter": bp.get("n_iter"), "search_rows": bp.get("search_rows")}


@task
def promote_model(verdict: dict) -> dict:
    """Step 4 — promote a passing candidate to production, or keep the current one."""
    logger = get_run_logger()
    if not verdict["passed"]:
        logger.warning(f"Candidate PR-AUC {verdict['candidate_pr']} did not beat the gate "
                       f"{verdict['gate']} — KEEPING current production model.")
        return {"promoted": False, **verdict}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(verdict["candidate_path"], PRODUCTION)
    # production_meta.json is the durable record of what's actually deployed.
    PROD_META.write_text(json.dumps({
        "test_pr_auc": verdict["candidate_pr"],
        "model": PRODUCTION.name,
        "params": verdict.get("candidate_params", {}),
        "n_iter": verdict.get("n_iter"),
        "search_rows": verdict.get("search_rows"),
    }, indent=2))
    logger.info(f"Promoted candidate (PR-AUC {verdict['candidate_pr']}) -> {PRODUCTION.name}")
    return {"promoted": True, **verdict}


@flow(name="retrain-loan-model")
def retrain_flow(sample: int = 120_000, n_iter: int = 25) -> dict:
    """The pipeline. Prefect runs the tasks in dependency order."""
    logger = get_run_logger()
    logger.info("===== Retrain pipeline started =====")
    raw = download_data()
    data = clean_data(raw)
    candidate = tune_model(data, sample, n_iter)
    verdict = validate_model(candidate)
    result = promote_model(verdict)
    logger.info(f"===== Retrain pipeline finished: {result} =====")
    return result


if __name__ == "__main__":
    if "serve" in sys.argv:
        # Attach a schedule and stay running (the Prefect equivalent of Celery Beat).
        # Cron: minute hour day month weekday. "0 3 * * 0" = 03:00 every Sunday.
        # timezone pins it to LOCAL Nairobi time (EAT, UTC+3) so 03:00 means 03:00 here.
        # NOTE: a scheduled run only FIRES if a persistent server is running
        # (`make prefect-server`); the ephemeral server can't schedule.
        from prefect.schedules import Cron
        retrain_flow.serve(
            name="weekly-retrain",
            schedules=[Cron("0 3 * * 0", timezone="Africa/Nairobi")],
        )
    else:
        retrain_flow()
