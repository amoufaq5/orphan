# Orphan — Repo Map & Module Guide

## Top Level
- **conf/** — All runtime configuration (YAML + lists)
  - `app.yaml` – API/runtime toggles, model+index paths, ports.
  - `data.yaml` – Data lake layout (raw/clean/corpus/out) & schemas.
  - `scrape.yaml` – Source toggles (pubmed, ctgov, openfda, dailymed, pmc_oa…), query terms, paging, caps.
  - `train_text.yaml` – TextLM pretrain config (tokenizer path, dataset, hparams).
  - `train_sft.yaml` – SFT config (task templates, sampling, hparams).
  - `disease_terms.txt` – Seed terms for scrapers.
- **scripts/** — Operator entrypoints
  - `run_local.sh` – One-shot runner (scrape → clean → train → rag → eval toggleable by flags).
- **src/** — All code
  - **api/**
    - `main.py` – FastAPI app factory (health, /infer, /rag endpoints).
    - `infer.py` – Model load + generate wrapper.
    - `offline_infer.py` – File/CLI batch inference for experiments.
    - `rag.py` – RAG HTTP endpoints using built FAISS index.
  - **chat/**
    - `patient_cli.py` – Local CLI for patient-style dialogue/testing.
  - **clean/**
    - `normalize.py` / `to_canonical.py` – Normalize each source to canonical schema.
    - `ctgov_flatten.py` – ClinicalTrials field flattening & joins.
    - `enrich_labels.py` / `label_sectioner.py` – Sectioning, weak labels, heuristics.
    - `make_corpus.py` – Builds training/eval corpus from canonical.
    - `build_index.py` – Text corpora for retrieval (doc store).
    - `run_canonical.py`, `run_enrich_labels.py` – Pipelines/CLIs.
  - **eval/**
    - `metrics.py` – Task metrics (exact-match, Rouge, F1, retrieval@k, MRR).
    - `predict_one.py`, `predict_batch.py` – Quick sanity & batch eval.
    - `visualize.py` – Plots/curves; debug samples.
    - `leaderboard.py` – Summarizes runs across seeds/checkpoints.
  - **models/**
    - **textlm/** – Base LM (≈ your 140M)
      - `dataset.py` – Streaming JSONL dataset, packing, masking.
      - `model.py` – Model def/wrapper (HF compatible).
      - `train_text.py` – Trainer (AMP, grad-accum, checkpointing).
      - `train_tokenizer.py` – Optional tokenizer from corpus shards.
      - `utils.py` – Collators, schedulers, logging hooks.
    - **sft/** – Supervised instruction fine-tuning
      - `train_sft.py` – SFT loop (LoRA/full-fine-tune per config).
      - `build_tasks.py`, `templates.py`, `safety_prompts.py`, `schema.py`, `dataset.py`.
    - **tokenizer/**
      - `build_corpus.py` – Tokenizer corpus assembly.
      - `train_tokenizer_spm.py` – SentencePiece training.
      - `wrap_hf_tokenizer.py`, `quick_encode.py`.
  - **ontology/**
    - `icd10.py`, `mesh.py`, `rxnorm.py`, `atc.py` – Code systems & mappings.
  - **rag/**
    - `chunk.py` – Chunking strategies (sliding, semantic).
    - `embed_faiss.py` – Embedding + FAISS index build.
    - `index.py` – Index I/O, metadata, shard mgmt.
    - `query.py`, `query_faiss.py` – Retrieval API + CLI.
    - `eval_retrieval.py` – R@k/MRR, latency measurements.
    - `build_index.py`, `run_build_index.py` – End-to-end build.
  - **scrapers/**
    - `base_scraper.py` – Sessions, retries, politeness, paging, caps.
    - `pubmed.py`, `ctgov.py`, `clinicaltrials.py`, `openfda_labels.py`,
      `dailymed.py`, `pmc_oa.py`, `registry.py`, `http.py`, `run.py`.
  - **utils/**
    - `config.py` – YAML loader & dataclass merges.
    - `logger.py` – Structured logging.
    - `io.py` – JSONL/Parquet sharding, gzip, path helpers.
    - `schemas.py` – Canonical record types.
    - `rate_limit.py` – Backoff, throttling.
    - `hashing.py`, `pii.py` – Dedup, PII masking.
- **requirements.txt**, **pyproject.toml**, **README.md**, **LICENSE**, **.gitignore**

## Reference Paths (suggested)
- Raw data: `data/raw/<source>/…`
- Clean/canonical: `data/clean/…`
- Corpus: `data/corpus/…`
- Indices: `data/index/faiss/…`
- Models: `out/models/textlm/`, `out/models/sft/`
- Logs & metrics: `out/logs/`, `out/metrics/`

## Common Commands
```bash
# Scrape (examples are toggled via conf/scrape.yaml)
python -m src.scrapers.run -c conf/scrape.yaml --sources pubmed,ctgov,openfda --workers 4

# Normalize → Canonical → Corpus
python -m src.clean.run_canonical   -c conf/data.yaml
python -m src.clean.run_enrich_labels -c conf/data.yaml
python -m src.clean.make_corpus     -c conf/data.yaml

# Tokenizer (optional if already trained)
python -m src.models.tokenizer.train_tokenizer_spm -c conf/train_text.yaml

# Base LM pretrain / continue-pretrain
python -m src.models.textlm.train_text -c conf/train_text.yaml

# SFT
python -m src.models.sft.train_sft -c conf/train_sft.yaml

# Build RAG index
python -m src.rag.run_build_index -c conf/app.yaml

# Retrieval test
python -m src.rag.query_faiss -q "adult with chest pain and wheeze..." -c conf/app.yaml

# Inference
python -m src.api.offline_infer --text "patient: ..." -c conf/app.yaml
python -m src.api.main  # serves /infer and /rag
