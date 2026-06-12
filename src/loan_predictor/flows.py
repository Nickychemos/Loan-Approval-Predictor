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
import shutil
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/hmda_multistate_2023.csv"
CLEAN = ROOT / "data/processed/hmda_clean.parquet"
CANDIDATE = ROOT / "models/loan_approval_model_candidate.joblib"
PRODUCTION = ROOT / "models/loan_approval_model_tuned.joblib"
BEST_PARAMS = ROOT / "reports/best_params.json"
PROD_META = ROOT / "reports/production_meta.json"
MIN_PR_AUC = 0.90   # absolute floor; the real gate is "beat current production"


@task(retries=2, retry_delay_seconds=15)
def clean_data() -> str:
    """Step 1 — rebuild the model-ready dataset. Retried twice if it fails."""
    logger = get_run_logger()
    logger.info("Cleaning raw HMDA data (dedup, feature-engineer, winsorise)...")
    from loan_predictor.clean import clean
    clean(str(RAW), str(CLEAN))
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
    candidate_pr = json.loads(BEST_PARAMS.read_text())["test_pr_auc"]
    prod_pr = 0.0
    if PROD_META.exists():
        prod_pr = json.loads(PROD_META.read_text()).get("test_pr_auc", 0.0)
    gate = max(MIN_PR_AUC, prod_pr)
    passed = candidate_pr >= gate
    logger.info(f"Candidate PR-AUC={candidate_pr}  |  gate(max floor/prod)={gate}  "
                f"=> {'PASS' if passed else 'REJECT'}")
    return {"candidate_path": candidate_path, "candidate_pr": candidate_pr,
            "gate": gate, "passed": passed}


@task
def promote_model(verdict: dict) -> dict:
    """Step 4 — promote a passing candidate to production, or keep the current one."""
    logger = get_run_logger()
    if not verdict["passed"]:
        logger.warning(f"Candidate PR-AUC {verdict['candidate_pr']} did not beat the gate "
                       f"{verdict['gate']} — KEEPING current production model.")
        return {"promoted": False, **verdict}
    shutil.copy(verdict["candidate_path"], PRODUCTION)
    PROD_META.write_text(json.dumps({"test_pr_auc": verdict["candidate_pr"],
                                     "model": PRODUCTION.name}, indent=2))
    logger.info(f"Promoted candidate (PR-AUC {verdict['candidate_pr']}) -> {PRODUCTION.name}")
    return {"promoted": True, **verdict}


@flow(name="retrain-loan-model")
def retrain_flow(sample: int = 120_000, n_iter: int = 25) -> dict:
    """The pipeline. Prefect runs the tasks in dependency order."""
    logger = get_run_logger()
    logger.info("===== Retrain pipeline started =====")
    data = clean_data()
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
