# Makefile for the Loan Approval Predictor (housing-backed)
# ML pipeline (Polars / scikit-learn / XGBoost) + Django REST Framework API on Postgres.

# Variables
PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
MANAGE  := $(PYTHON) webapp/manage.py
ML      := PYTHONPATH=src $(PYTHON) -m loan_predictor
PORT    := 8001    # 8000 is often taken by other local projects; override: make run PORT=xxxx
RAW     := data/raw/hmda_multistate_2023.csv
HMDA_URL := https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=MD,VA,CO,OR,TN,MO&years=2023&actions_taken=1,3

.DEFAULT_GOAL := help

# ---------- Help ----------
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------- Environment ----------
venv: ## Create the virtualenv if missing
	test -d .venv || python3.12 -m venv .venv

install: venv ## Install Python dependencies into the venv
	$(PIP) install -q -r requirements.txt

# ---------- Data & ML pipeline ----------
data: ## Download real HMDA training data to data/raw/
	mkdir -p data/raw
	curl -L "$(HMDA_URL)" -o $(RAW)

clean-data: ## Clean raw HMDA -> tidy parquet (dedup + feature engineering)
	$(ML).clean --input $(RAW)

train: ## Compare 6 algorithms and save the best model
	$(ML).train

evaluate: ## Cross-validate + calibrate the winner
	$(ML).evaluate

tune: ## Hyperparameter-tune XGBoost and save the production model
	$(ML).tune

ml: clean-data train tune ## Run the full ML pipeline (clean -> train -> tune)

# ---------- Orchestration (Prefect) ----------
# PREFECT_..._TIMEOUT gives the first-run ephemeral server time to migrate its DB.
PREFECT_ENV := PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=120

flow: ## Run the retraining flow once (Prefect)
	$(PREFECT_ENV) $(ML).flows

prefect-server: ## Start the persistent Prefect server + UI (http://localhost:4200) — needed for schedules
	$(PYTHON) -m prefect server start

flow-serve: ## Register the schedule (Sundays 03:00 EAT) — run `make prefect-server` first
	$(PREFECT_ENV) $(ML).flows serve

# ---------- Database (Docker Postgres) ----------
db-up: ## Start the Postgres container (waits until healthy)
	docker compose up -d --wait postgres

db-down: ## Stop the Postgres container
	docker compose down

db-logs: ## Tail Postgres logs
	docker compose logs -f postgres

# ---------- Django / DRF ----------
migrate: ## Make and apply migrations
	$(MANAGE) makemigrations
	$(MANAGE) migrate

migrations: ## Create new migrations only
	$(MANAGE) makemigrations

superuser: ## Create an admin user
	$(MANAGE) createsuperuser

server: ## Start the API / dev server
	$(MANAGE) runserver $(PORT)

run: db-up migrate server ## Start Postgres, migrate, then run the server

shell: ## Open the Django shell
	$(MANAGE) shell

dbshell: ## Open the Postgres shell
	$(MANAGE) dbshell

reset-db: ## Flush all data and re-migrate (WARNING: destroys data)
	$(MANAGE) flush --no-input
	$(MANAGE) migrate

# ---------- Testing & quality ----------
test: ## Run the API test suite (model + endpoints)
	$(MANAGE) test applications -v2

check: ## Django system checks + missing-migration check
	$(MANAGE) check
	$(MANAGE) makemigrations --check --dry-run

clean: ## Remove __pycache__ and .pyc files
	find . -path ./.venv -prune -o -type f -name "*.pyc" -exec rm -f {} +
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} +

# ---------- Full setup ----------
setup: install db-up migrate ## One-shot setup: deps + Postgres + migrations
	@echo "Setup complete. Run 'make superuser' then 'make run'."

# ---------- Aliases ----------
s: server     ## Alias for server
m: migrate    ## Alias for migrate
t: test       ## Alias for test

.PHONY: help venv install data clean-data train evaluate tune ml flow prefect-server flow-serve db-up db-down db-logs \
        migrate migrations superuser server run shell dbshell reset-db test check clean setup s m t
