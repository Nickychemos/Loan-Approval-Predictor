"""A tiny flow used only to prove the Prefect Cloud deploy → run → dashboard pipe.

It does no ML and needs no data — it just logs a message, so it runs in seconds
on Prefect Cloud's free managed compute. Once this works end-to-end, we know the
GitHub connection and Cloud execution are good, and we can deploy the real
retrain flow against our own (Oracle) compute.
"""
from prefect import flow, get_run_logger


@flow(name="hello-cloud")
def hello_world(name: str = "Nicky") -> str:
    logger = get_run_logger()
    logger.info(f"Hello {name}! This flow ran on Prefect Cloud's managed compute.")
    logger.info("If you can read this in the Cloud dashboard, the pipe works.")
    return f"Hello {name}"


if __name__ == "__main__":
    hello_world()
