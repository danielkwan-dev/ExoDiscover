.PHONY: install ingest train eval test lint serve web docker
install:
	pip install -e ".[dev]" fastapi "uvicorn[standard]" python-multipart httpx
	cd web && npm install

ingest:
	exo ingest

train:
	exo train

eval:
	exo eval

test:
	pytest
	cd web && npm run test

lint:
	ruff check .
	mypy
	cd web && npm run lint

serve:
	uvicorn api.main:app --reload --port 8000

web:
	cd web && npm run dev

docker:
	docker compose up --build
