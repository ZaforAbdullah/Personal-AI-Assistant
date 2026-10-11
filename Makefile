# Convenience targets; every one of these is also spelled out in the README.

.PHONY: up down logs test test-backend test-frontend lint dev-backend dev-frontend db

up:            ## build and start the whole stack (db, backend, frontend)
	docker compose up --build -d

down:          ## stop the stack (add -v yourself to drop the database volume)
	docker compose down

logs:
	docker compose logs -f backend frontend

db:            ## only Postgres, for running backend/frontend on the host
	docker compose up -d db

dev-backend:   ## hot-reloading API on :8000 (needs backend/.venv, see README)
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

dev-frontend:  ## Next.js dev server on :3000
	cd frontend && pnpm dev

test: test-backend test-frontend

test-backend:
	cd backend && .venv/bin/pytest -q

test-frontend:
	cd frontend && pnpm test

lint:
	cd backend && .venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests
	cd frontend && pnpm lint && pnpm typecheck
