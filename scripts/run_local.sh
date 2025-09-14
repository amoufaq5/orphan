#!/usr/bin/env bash
set -euo pipefail

# 0) create dirs (idempotent)
mkdir -p data/raw data/cleaned data/canonical data/shards out

# 1) (optional) FIRST BATCH — comment out any sources you don’t want
# python -m src.scrapers.run --config conf/scrape.yaml --sources dailymed_spls openfda_labels pubmed_abstracts pmc_open_access clinicaltrials
# python -m src.clean.run_canonical --data conf/data.yaml --prefix dailymed_spls openfda_labels pubmed_abstracts pmc_open_access clinicaltrials
# python -m src.clean.run_enrich_labels --data conf/data.yaml

# 2) build RAG index from whatever canonical files exist
python -m src.rag.run_build_index --config conf/app.yaml

# 3) tokenizer (build+train) — requires sentencepiece already installed locally
# python -m src.models.tokenizer.build_corpus --config conf/train_text.yaml
# python -m src.models.tokenizer.train_tokenizer_spm --config conf/train_text.yaml

# 4) run offline inference (no server, no internet)
python -m src.api.offline_infer --config conf/app.yaml --persona doctor \
  --instruction "List key contraindications and serious warnings for ibuprofen." || true
