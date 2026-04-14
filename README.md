# SEBI Orders RAG

A retrieval-augmented question-answering system for SEBI enforcement and adjudication orders. The project combines a portable downloader, a multi-phase ingestion pipeline, PostgreSQL + pgvector retrieval, a FastAPI-backed chat surface for grounded answers over SEBI order data, and an evaluation toolkit for replay, red-team, and release-gate runs.

Intended public repo: [`hariharan077/sebi-orders-rag`](https://github.com/hariharan077/sebi-orders-rag)

This repo is intended to be forkable and runnable on a developer's own machine. A fork should work with local PostgreSQL, a locally downloaded SEBI corpus, and the fork owner's own OpenAI API key in `.env`.

## What This Repo Shows

- A portable downloader that can fetch SEBI order PDFs by category on demand.
- A structured ingestion pipeline for manifests, extraction, chunking, embeddings, and chat.
- Adaptive routing between exact lookup, hierarchical RAG, memory-scoped follow-ups, and current-information paths.
- An internal evaluation engine for dataset assembly, replay/live benchmarking, red-team coverage, and release gating.
- Regression and routing tests backed by small committed fixtures instead of bulky local artifacts.

## Stack

- Python
- PostgreSQL with `pgvector` and `pg_trgm`
- FastAPI
- OpenAI embeddings/chat APIs

## Project Layout

```text
app/                    application code, routing, retrieval, chat, evaluation modules
scripts/                downloader, phase CLIs, and eval/release-gate entrypoints
tests/                  regression, routing, and evaluation tests
tests/fixtures/         committed minimal control-pack and eval-dump fixtures
requirements-sebi-orders-rag.txt
```

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/hariharan077/sebi-orders-rag.git
cd sebi-orders-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-sebi-orders-rag.txt pytest
```

### 2. Configure local environment

```bash
cp .env.example .env
```

Set your own values in `.env`.

- `SEBI_ORDERS_RAG_DB_DSN` should point to your local PostgreSQL database.
- `SEBI_ORDERS_RAG_OPENAI_API_KEY` must be your own OpenAI API key.
- `SEBI_ORDERS_RAG_DATA_ROOT` should point at your local downloaded corpus directory.
- `SEBI_ORDERS_RAG_CONTROL_PACK_ROOT` is optional. Leave it blank unless you generate a control pack locally.

No real API keys are committed to this repo. Anyone who forks it must add their own key before running embedding or chat workflows.

### 3. Run the app locally

Once PostgreSQL is running and `.env` is configured, start the local FastAPI app:

```bash
python -m app.sebi_orders_rag.api.phase4_app
```

The app uses `SEBI_ORDERS_RAG_PHASE4_APP_HOST` and `SEBI_ORDERS_RAG_PHASE4_APP_PORT` from `.env`. The CLI chat entrypoint is also available for quick local validation:

```bash
python scripts/sebi_orders_phase4_chat.py \
  --query "What did SEBI direct in the Vishvaraj Environment Limited matter?"
```

## Demo Path

These commands are a lightweight smoke path from a fresh clone:

```bash
python scripts/download_orders_portable.py examples
python scripts/sebi_orders_phase1.py --help
python scripts/sebi_orders_phase4_chat.py --help
python scripts/sebi_eval_run.py --help
python -m pytest \
  tests/sebi_orders_rag/test_control_loader.py \
  tests/sebi_orders_rag/test_router.py \
  tests/sebi_orders_rag/test_eval_equivalent_routes.py \
  tests/sebi_orders_rag/test_release_gate.py
```

To download a small local slice of the corpus:

```bash
python scripts/download_orders_portable.py backfill \
  --category orders-of-sat \
  --output-dir ./sebi-orders-pdfs
```

## Full Pipeline

### 1. Download corpus data locally

```bash
python scripts/download_orders_portable.py backfill --output-dir ./sebi-orders-pdfs
```

You can also fetch only selected categories:

```bash
python scripts/download_orders_portable.py backfill \
  --category orders-of-sat \
  --category orders-of-ao \
  --output-dir ./sebi-orders-pdfs
```

### 2. Initialize and ingest

```bash
python scripts/sebi_orders_phase1.py --init-db --apply
python scripts/sebi_orders_phase2.py --run-migration --apply
python scripts/sebi_orders_phase3_embed.py --run-migration --apply
```

### 3. Query the system

```bash
python scripts/sebi_orders_phase4_chat.py \
  --query "What did SEBI direct in the Vishvaraj Environment Limited matter?"
```

### 4. Run evaluation and release-gate checks

```bash
python scripts/sebi_eval_run.py --help
python scripts/sebi_eval_release_gate.py --help
```

The evaluation commands can build datasets from committed control-pack fixtures, replay stored cases, and optionally layer in local failure dumps or red-team cases without committing generated outputs.

## Optional Control-Pack Workflow

The repo does not commit timestamped `artifacts/` outputs. If you want local eval/regression packs, generate them on your machine:

```bash
python scripts/build_sebi_control_pack.py
python scripts/sebi_orders_phase4_eval.py --run-eval-queries --run-regressions
```

If you generate a control pack, either let the app discover the latest pack under `artifacts/` automatically or point `SEBI_ORDERS_RAG_CONTROL_PACK_ROOT` at it explicitly.

## Notes

- `sebi-orders-pdfs/` is intentionally excluded from version control. Use the downloader to fetch data locally.
- `artifacts/` is also excluded. Those are local/generated outputs, not source-of-truth project files.
- Local databases, virtualenvs, caches, and `.env` files are excluded as well. The committed repo is limited to source, scripts, tests, and small fixtures.
- Committed fixtures under `tests/fixtures/` are only the minimum needed to keep a few regression tests reproducible in public.
