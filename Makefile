PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install demo-data train migrate seed backend frontend test smoke lint docker-up

install:
	$(PIP) install -e ".[dev]"
	cd frontend && npm install

demo-data:
	$(PYTHON) -m ml.generate_data

train:
	$(PYTHON) scripts/train_all.py

migrate:
	$(PYTHON) -m alembic upgrade head

seed:
	$(PYTHON) scripts/seed_demo.py

backend:
	$(PYTHON) -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest tests -q

smoke:
	$(PYTHON) scripts/run_smoke_tests.py

lint:
	$(PYTHON) -m ruff check backend ml scripts tests
	cd frontend && npm run build

docker-up:
	docker compose up --build
