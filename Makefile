.PHONY: install run test test-fast docker-up docker-down docker-logs pull-model

install:
	pip install -e ".[dev]"

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -v -x --ignore=tests/test_rca_agent.py --ignore=tests/test_report_agent.py --ignore=tests/test_routes.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f api

pull-model:
	docker compose exec ollama ollama pull tinyllama # Put mistral if you have enough RAM you will have better results
