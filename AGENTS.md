# Repository Guidelines

## Project Structure & Module Organization
- `Dockerfile`, `docker-compose.yml`: Neo4j container setup for the temporal knowledge graph.
- `neo4j-entrypoint.sh`: Wrapper entrypoint that waits for Neo4j and applies schema.
- `schema.cypher`: Cypher schema/index definitions (currently empty; see `PLAN.md` draft).
- `settings.py`: Central configuration (models, embedding dims, entity types).
- `prompts.py`: LLM prompt templates for extraction and RAG answering.
- `PLAN.md`: MVP roadmap and schema draft.
- `.env`: Local runtime configuration (do not commit secrets).

## Build, Test, and Development Commands
- `docker compose up --build`: Build and run the Neo4j service with schema application.
- `docker compose down`: Stop and remove the service containers.
- `cypher-shell -a bolt://localhost:7688 -u neo4j -p passworty`: Connect to Neo4j from the host (ports mapped in `docker-compose.yml`).

## Coding Style & Naming Conventions
- Python: follow PEP 8 (4-space indentation), `snake_case` for variables/functions, `UPPER_SNAKE_CASE` for constants (see `settings.py`).
- Shell: keep `bash` scripts POSIX-friendly and strict (`set -euo pipefail` already used).
- Cypher: use uppercase for keywords and descriptive index/constraint names.

## Testing Guidelines
- No automated tests are present yet. If you add tests, document them here and prefer a standard framework (e.g., `pytest`) with naming like `test_*.py`.
- Until tests exist, verify changes by:
  - running `docker compose up --build`
  - checking Neo4j schema via `cypher-shell` after startup.

## Commit & Pull Request Guidelines
- Commit history is minimal and does not establish conventions. Use concise, imperative messages (e.g., “Add schema constraints”).
- PRs should include:
  - a short summary of what changed and why,
  - any relevant schema or prompt changes,
  - verification steps (commands run and results).

## Security & Configuration Tips
- Store credentials and API keys in `.env`; never commit secrets.
- If you change Neo4j credentials, update both `.env` (if used) and `docker-compose.yml` to keep them aligned.
