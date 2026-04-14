# SEBI Orders RAG

A retrieval-augmented question-answering system for SEBI enforcement and adjudication orders. The project combines a portable downloader, a multi-phase ingestion pipeline, PostgreSQL + pgvector retrieval, and a FastAPI-backed chat surface for grounded answers over SEBI order data.

Intended public repo: [`hariharan077/sebi-orders-rag`](https://github.com/hariharan077/sebi-orders-rag)

## What This Repo Shows

- A portable downloader that can fetch SEBI order PDFs by category on demand.
- A structured ingestion pipeline for manifests, extraction, chunking, embeddings, and chat.
- Adaptive routing between exact lookup, hierarchical RAG, memory-scoped follow-ups, and current-information paths.
- Regression and routing tests backed by small committed fixtures instead of bulky local artifacts.

## Stack

- Python
- PostgreSQL with `pgvector` and `pg_trgm`
- FastAPI
- OpenAI embeddings/chat APIs

## Project Layout

```text
app/                    application code, schemas, SQL, retrieval, chat routes
scripts/                downloader plus phase-oriented CLI entrypoints
tests/                  regression and routing tests
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

## Demo Path

These commands are a lightweight smoke path from a fresh clone:

```bash
python scripts/download_orders_portable.py examples
python scripts/sebi_orders_phase1.py --help
python scripts/sebi_orders_phase4_chat.py --help
python -m pytest \
  tests/sebi_orders_rag/test_control_loader.py \
  tests/sebi_orders_rag/test_router.py \
  tests/sebi_orders_rag/test_eval_equivalent_routes.py
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
- Committed fixtures under `tests/fixtures/` are only the minimum needed to keep a few regression tests reproducible in public.
