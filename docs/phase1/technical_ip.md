# Technical IP Dossier

## 1. Custom LLM Architecture Specifications

- **Model family:** `src/models/textlm/architecture.py` and `src/models/textlm/training/trainer.py` define a 7B-parameter transformer with rotary positional embeddings, gated feed-forward modules, and FlashAttention 2 kernels.
- **Tokenizer:** `src/models/tokenizer/medical_tokenizer.py` packages a 200K-entry vocabulary blended from SNOMED CT, RxNorm, UK NHS triage phrasing, and USMLE corpora. The tokenizer ships with adaptive domain-specific merges and subword annotations for medication dosages.
- **Multimodal fusion:** `src/models/multimodal/vision/medical_vision.py` integrates ResNet50/ViT backbones with cross-attention bridges (`VisionTextFusionBlock`) enabling late-fusion reasoning across radiology reports and imaging tensors.
- **Optimization:** H100-ready training loops (`conf/h100_config.yaml`, `scripts/training/train_multimodal.sh`) utilize DeepSpeed ZeRO-3 partitioning, mixed-precision BF16, parameter-efficient adapters, and gradient checkpointing.

## 2. Medical Protocol Implementations

- **WHAM/WWHAM:** `src/protocols/wham/protocol.py` implements full WWHAM flow (Who, What, How, Action, Medication, Monitoring) with dataclasses for patient context, symptom presentation, and safety netting. Integrated red flag detection references `src/protocols/safety/red_flags.py`.
- **AS METHOD:** `src/protocols/as_method/protocol.py` covers Age, Self-care, Medication, Extra, Time, History, Other, Danger with stateful question flow and triage branching.
- **Triage Orchestration:** `src/chat/triage.py` fuses AS METHOD + WWHAM outputs to determine OTC vs referral pathways; `src/chat/patient_interface.py` surfaces both protocols in conversational UX.

## 3. Training Methodologies

- **Data preprocessing:** Scraper outputs in `src/scrapers/` feed canonicalization pipelines (`src/clean/to_canonical.py`, `src/clean/normalize.py`), deduplicated through `src/utils/io.py` shard writers.
- **Curriculum:** Multi-stage script `scripts/training/train_multimodal.sh` orchestrates tokenizer pretraining → base language model SFT → safety alignment → multimodal tuning.
- **Evaluation:** `src/eval/` provides gold benchmarks (USMLE QA, red flag guards) with metrics aggregated via `src/eval/eval_gold.py`.
- **Continuous integration:** GitHub Actions templates (see `.github/workflows/`) run lint, unit tests, and doc build; model checkpoints archived per `conf/data.yaml` policies.

## 4. Development Logs

- **Architecture decisions:** Documented in `docs/architecture/` ADRs capturing protocol integration, ingestion design, and deployment topology.
- **Scraper evolution:** `docs/data/ingestion_log.md` tracks new sources (ChemBL, ClinVar, MedlinePlus, CDC, NIH wellness, Semantic Scholar, Europe PMC, BioRxiv, PLOS, CME, ACC, DrugBank, RxNorm) with API status, rate limits, and retry logic.
- **Experiment tracking:** Weights & Biases project IDs configured in `conf/tracking.yaml`; each fine-tuning run pushes metrics (`loss`, `safety_accuracy`) linked to dataset snapshots.
- **Release notes:** `docs/release/CHANGELOG.md` enumerates Phase 0/Phase 1 increments, including WWHAM enhancements, scraper expansions, and governance documentation.
