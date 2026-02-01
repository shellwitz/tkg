# TKG RAG

Temporal Knowledge Graph RAG pipeline built on Neo4j, with LLM-based entity/relation extraction, vector search, and a Cypher agent for query answering. The repo includes ingestion scripts, retrieval/answering code, and a Streamlit UI for inspection.

## What this project does
- Ingests text (e.g., ECT-QA corpus) into a temporal knowledge graph (entities + relations with timestamps).
- Builds vector indexes in Neo4j for chunk and relation embeddings.
- Retrieves relevant chunks and edges via vector search + graph scoring.
- Generates answers with an LLM using the retrieved context.
- Provides a Cypher agent that can plan and execute read-only queries.

## Repo layout
- `tkg_rag/` core library (ingest, retrieve, answer, cypher agent).
- `scripts/` ingestion, evaluation, and Streamlit inspection UI.
- `schema.cypher` Neo4j schema (constraints, full-text, range indexes).
- `neo4j-entrypoint.sh` applies schema and adds vector indexes at container start.
- `ect-qa/` dataset and questions (expected by scripts).
- `tests/` unit tests.

## Requirements
- Python 3.10+ (tested with 3.13 based on compiled cache files).
- Docker + Docker Compose.
- Neo4j (started via `docker-compose.yml`).
- LLM and embedding API access (see `.env.example`).

## Setup
1) Create a virtual environment and install deps:
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment variables:
```bash
cp .env.example .env
```
Then edit `.env` with your LLM and embedding provider credentials. Pay attention to:
- `LLM_MODEL` and `MODEL_API_KEY` for chat/completions.
- `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, and `EMBEDDING_DIM` (must match model output dimension).
- Neo4j ports and credentials if you change defaults.

3) Start Neo4j (schema + vector indexes are applied automatically):
```bash
docker compose up -d --build
```

## Ingest data
The ingestion scripts expect the ECT-QA corpus file at `ect-qa/extracted/corpus/base.jsonl`.

Examples:
```bash
# Ingest a single example
python scripts/ingest_test.py

# Fresh container + rebuild, then ingest stock-code subset
python scripts/ingest_test.py -fb -q

# Ingest the full corpus
python scripts/ingest_test.py -a
```

## Retrieve + answer
```bash
# Standard RAG (vector + edge retrieval)
python scripts/answer_test.py -q "What happened in 2020 Q1 related to Crocs?"

# Vector-only retrieval
python scripts/answer_test.py -m vec_search_only -qs
# -qs for answering question to the stock-code subset that got inserted with python scripts/ingest_test.py -q

# Cypher agent mode
python scripts/answer_test.py -m cypher_agent -qs
```
The output of the answer_test.py -qs is stored in ../tkg_eval such that it can be evaluated afterwards.

## Streamlit inspection UI
```bash
make web_show
```
This runs `scripts/streamlit_chunk_retrieval.py` and shows RAG results and Cypher agent outputs side-by-side.

## Tests
```bash
make test
```

## Notes
- Neo4j schema is defined in `schema.cypher`. The entrypoint appends vector indexes based on `EMBEDDING_DIM`.
- The Cypher agent uses DB introspection plus the schema for constrained query generation.
