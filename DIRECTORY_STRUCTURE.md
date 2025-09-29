# Orphan Medical AI Platform - Complete Directory Structure

## Root Directory Structure
```
orphan/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── gradio_app.py
├── DIRECTORY_STRUCTURE.md
│
├── conf/                           # Configuration files
│   ├── app.yaml                   # Main application config
│   ├── data.yaml                  # Data pipeline config
│   ├── scrape.yaml               # Scraper configurations
│   ├── train_text.yaml           # Base LM training config
│   ├── train_sft.yaml            # SFT training config
│   ├── train_multimodal.yaml     # Multimodal training config
│   ├── tokenizer.yaml            # Tokenizer config
│   ├── model_config.yaml         # Model architecture config
│   ├── kaggle_catalog.yaml       # Kaggle datasets config
│   ├── snomed_config.yaml        # SNOMED CT configuration
│   ├── h100_config.yaml          # H100 GPU optimization config
│   └── disease_terms.txt         # Medical terms for scraping
│
├── scripts/                       # Operational scripts
│   ├── setup/
│   │   ├── install_dependencies.sh
│   │   ├── setup_h100.sh
│   │   └── download_snomed.sh
│   ├── data/
│   │   ├── run_scrapers.sh
│   │   ├── process_kaggle.sh
│   │   └── build_corpus.sh
│   ├── training/
│   │   ├── train_tokenizer.sh
│   │   ├── train_base_model.sh
│   │   ├── train_sft.sh
│   │   └── train_multimodal.sh
│   ├── deployment/
│   │   ├── deploy_api.sh
│   │   ├── deploy_frontend.sh
│   │   └── health_check.sh
│   └── run_local.sh              # Main runner script
│
├── src/                          # Source code
│   ├── __init__.py
│   │
│   ├── api/                      # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app factory
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py        # Health endpoints
│   │   │   ├── inference.py     # Text inference
│   │   │   ├── multimodal.py    # Multimodal inference
│   │   │   ├── rag.py          # RAG endpoints
│   │   │   ├── patient.py      # Patient-specific endpoints
│   │   │   └── admin.py        # Admin endpoints
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py         # Authentication
│   │   │   ├── rate_limit.py   # Rate limiting
│   │   │   └── logging.py      # Request logging
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py     # Pydantic request models
│   │   │   └── responses.py    # Pydantic response models
│   │   └── dependencies.py     # FastAPI dependencies
│   │
│   ├── frontend/                # React + TypeScript frontend
│   │   ├── public/
│   │   │   ├── index.html
│   │   │   └── favicon.ico
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── Chat/
│   │   │   │   ├── Patient/
│   │   │   │   ├── Medical/
│   │   │   │   └── Common/
│   │   │   ├── pages/
│   │   │   ├── hooks/
│   │   │   ├── services/
│   │   │   ├── types/
│   │   │   ├── utils/
│   │   │   ├── App.tsx
│   │   │   └── index.tsx
│   │   ├── package.json
│   │   └── tsconfig.json
│   │
│   ├── models/                  # AI/ML Models
│   │   ├── __init__.py
│   │   │
│   │   ├── textlm/             # Base Language Model
│   │   │   ├── __init__.py
│   │   │   ├── architecture/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── transformer.py      # Custom transformer
│   │   │   │   ├── attention.py        # Medical attention blocks
│   │   │   │   ├── embeddings.py       # Medical embeddings
│   │   │   │   └── heads.py           # Task-specific heads
│   │   │   ├── training/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py         # H100-optimized trainer
│   │   │   │   ├── dataset.py         # Medical dataset loader
│   │   │   │   ├── collator.py        # Data collation
│   │   │   │   └── callbacks.py       # Training callbacks
│   │   │   ├── inference/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── generator.py       # Text generation
│   │   │   │   └── batch_infer.py     # Batch inference
│   │   │   └── utils.py
│   │   │
│   │   ├── multimodal/         # Multimodal Models
│   │   │   ├── __init__.py
│   │   │   ├── vision/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── encoder.py         # Vision encoder
│   │   │   │   ├── processor.py       # Image processing
│   │   │   │   └── medical_vision.py  # Medical image analysis
│   │   │   ├── fusion/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cross_attention.py # Cross-modal attention
│   │   │   │   └── fusion_layers.py   # Fusion mechanisms
│   │   │   ├── training/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── trainer.py         # Multimodal trainer
│   │   │   │   └── dataset.py         # Multimodal dataset
│   │   │   └── inference/
│   │   │       ├── __init__.py
│   │   │       └── multimodal_infer.py
│   │   │
│   │   ├── sft/                # Supervised Fine-Tuning
│   │   │   ├── __init__.py
│   │   │   ├── medical_sft.py         # Medical SFT trainer
│   │   │   ├── templates.py           # Prompt templates
│   │   │   ├── safety_prompts.py      # Safety prompts
│   │   │   ├── protocol_prompts.py    # WHAM/AS METHOD prompts
│   │   │   └── dataset.py             # SFT dataset
│   │   │
│   │   └── tokenizer/          # Tokenizer
│   │       ├── __init__.py
│   │       ├── medical_tokenizer.py   # Medical vocabulary tokenizer
│   │       ├── build_vocab.py         # Vocabulary builder
│   │       └── train_tokenizer.py     # Tokenizer training
│   │
│   ├── data/                   # Data Processing
│   │   ├── __init__.py
│   │   │
│   │   ├── scrapers/           # Enhanced Scrapers
│   │   │   ├── __init__.py
│   │   │   ├── base/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── scraper.py         # Base scraper class
│   │   │   │   ├── session.py         # Session management
│   │   │   │   └── rate_limiter.py    # Rate limiting
│   │   │   ├── medical/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── pubmed_full.py     # Full PubMed articles
│   │   │   │   ├── pmc_scraper.py     # PMC full text
│   │   │   │   ├── clinicaltrials.py  # ClinicalTrials.gov
│   │   │   │   ├── cochrane.py        # Cochrane reviews
│   │   │   │   ├── nice_scraper.py    # NICE guidelines
│   │   │   │   └── bmj_scraper.py     # BMJ articles
│   │   │   ├── kaggle/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── downloader.py      # Kaggle dataset downloader
│   │   │   │   └── processor.py       # Kaggle data processor
│   │   │   └── registry.py            # Scraper registry
│   │   │
│   │   ├── processing/         # Data Processing
│   │   │   ├── __init__.py
│   │   │   ├── cleaning/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── text_cleaner.py    # Text cleaning
│   │   │   │   ├── medical_cleaner.py # Medical text processing
│   │   │   │   └── image_processor.py # Medical image processing
│   │   │   ├── normalization/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── canonical.py       # Canonical format
│   │   │   │   └── schema_mapper.py   # Schema mapping
│   │   │   └── augmentation/
│   │   │       ├── __init__.py
│   │   │       ├── text_augment.py    # Text augmentation
│   │   │       └── image_augment.py   # Image augmentation
│   │   │
│   │   └── loaders/            # Data Loaders
│   │       ├── __init__.py
│   │       ├── medical_loader.py      # Medical dataset loader
│   │       ├── multimodal_loader.py   # Multimodal loader
│   │       └── streaming_loader.py    # Streaming data loader
│   │
│   ├── ontology/               # Medical Ontologies
│   │   ├── __init__.py
│   │   ├── snomed/
│   │   │   ├── __init__.py
│   │   │   ├── loader.py              # SNOMED CT loader
│   │   │   ├── mapper.py              # Disease-symptom mapping
│   │   │   ├── search.py              # SNOMED search
│   │   │   └── validator.py           # Code validation
│   │   ├── icd10/
│   │   │   ├── __init__.py
│   │   │   └── mapper.py              # ICD-10 mapping
│   │   ├── mesh/
│   │   │   ├── __init__.py
│   │   │   └── mapper.py              # MeSH mapping
│   │   └── unified/
│   │       ├── __init__.py
│   │       └── ontology_bridge.py     # Unified ontology interface
│   │
│   ├── rag/                    # Retrieval-Augmented Generation
│   │   ├── __init__.py
│   │   ├── indexing/
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py             # Medical text chunking
│   │   │   ├── embedder.py            # Medical embeddings
│   │   │   ├── faiss_index.py         # FAISS indexing
│   │   │   └── chromadb_index.py      # ChromaDB indexing
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── retriever.py           # Document retrieval
│   │   │   ├── reranker.py            # Result reranking
│   │   │   └── medical_search.py      # Medical-specific search
│   │   └── generation/
│   │       ├── __init__.py
│   │       ├── rag_generator.py       # RAG generation
│   │       └── context_manager.py     # Context management
│   │
│   ├── protocols/              # Medical Protocols
│   │   ├── __init__.py
│   │   ├── wham/
│   │   │   ├── __init__.py
│   │   │   ├── protocol.py            # WHAM implementation
│   │   │   ├── validator.py           # WHAM validation
│   │   │   └── templates.py           # WHAM templates
│   │   ├── as_method/
│   │   │   ├── __init__.py
│   │   │   ├── protocol.py            # AS METHOD implementation
│   │   │   └── validator.py           # AS METHOD validation
│   │   └── safety/
│   │       ├── __init__.py
│   │       ├── red_flags.py           # Red flag detection
│   │       ├── contraindications.py   # Drug contraindications
│   │       └── safety_checker.py      # Safety validation
│   │
│   ├── chat/                   # Patient Interface
│   │   ├── __init__.py
│   │   ├── patient_interface.py       # Natural language interface
│   │   ├── conversation_manager.py    # Conversation state
│   │   ├── symptom_analyzer.py        # Symptom analysis
│   │   └── recommendation_engine.py   # Treatment recommendations
│   │
│   ├── evaluation/             # Model Evaluation
│   │   ├── __init__.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── medical_metrics.py     # Medical-specific metrics
│   │   │   ├── safety_metrics.py      # Safety evaluation
│   │   │   └── multimodal_metrics.py  # Multimodal evaluation
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   ├── usmle_eval.py          # USMLE evaluation
│   │   │   ├── medical_qa_eval.py     # Medical QA evaluation
│   │   │   └── clinical_eval.py       # Clinical evaluation
│   │   └── visualization/
│   │       ├── __init__.py
│   │       ├── plots.py               # Evaluation plots
│   │       └── reports.py             # Evaluation reports
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── loader.py              # Configuration loader
│       │   └── validator.py           # Config validation
│       ├── logging/
│       │   ├── __init__.py
│       │   ├── logger.py              # Structured logging
│       │   └── medical_logger.py      # Medical event logging
│       ├── io/
│       │   ├── __init__.py
│       │   ├── file_handler.py        # File operations
│       │   ├── compression.py         # Data compression
│       │   └── streaming.py           # Streaming I/O
│       ├── security/
│       │   ├── __init__.py
│       │   ├── pii_scrubber.py        # PII removal
│       │   ├── encryption.py          # Data encryption
│       │   └── audit.py               # Audit logging
│       └── monitoring/
│           ├── __init__.py
│           ├── health_check.py        # System health
│           ├── performance.py         # Performance monitoring
│           └── alerts.py              # Alert system
│
├── data/                       # Data Storage
│   ├── raw/                   # Raw scraped data
│   │   ├── pubmed/
│   │   ├── pmc/
│   │   ├── clinicaltrials/
│   │   ├── cochrane/
│   │   ├── nice/
│   │   ├── bmj/
│   │   └── kaggle/
│   ├── processed/             # Processed data
│   │   ├── text/
│   │   ├── images/
│   │   └── multimodal/
│   ├── corpus/                # Training corpus
│   │   ├── pretrain/
│   │   ├── sft/
│   │   └── multimodal/
│   ├── ontologies/            # Medical ontologies
│   │   ├── snomed/
│   │   ├── icd10/
│   │   └── mesh/
│   └── index/                 # Search indices
│       ├── faiss/
│       └── chromadb/
│
├── models/                    # Trained Models
│   ├── tokenizer/            # Tokenizer models
│   ├── textlm/              # Base language models
│   ├── sft/                 # Fine-tuned models
│   ├── multimodal/          # Multimodal models
│   └── embeddings/          # Embedding models
│
├── outputs/                   # Training Outputs
│   ├── logs/                 # Training logs
│   ├── checkpoints/          # Model checkpoints
│   ├── metrics/              # Evaluation metrics
│   └── reports/              # Training reports
│
├── tests/                     # Test Suite
│   ├── __init__.py
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── e2e/                  # End-to-end tests
│   └── fixtures/             # Test fixtures
│
└── docs/                      # Documentation
    ├── api/                  # API documentation
    ├── models/               # Model documentation
    ├── deployment/           # Deployment guides
    └── medical/              # Medical protocol docs
```

## Key Features of This Structure

### 1. **Multimodal Architecture**
- Separate vision and fusion modules
- Cross-modal attention mechanisms
- Medical image processing capabilities

### 2. **H100 Optimization**
- Dedicated H100 configuration
- Optimized training pipelines
- Memory-efficient implementations

### 3. **SNOMED CT Integration**
- Complete SNOMED CT loader and mapper
- Disease-symptom relationship mapping
- Unified ontology bridge

### 4. **Enhanced Scrapers**
- Full article extraction (not just abstracts)
- Multiple medical journal sources
- Kaggle dataset integration

### 5. **Medical Protocols**
- WHAM protocol implementation
- AS METHOD protocol support
- Safety and red flag detection

### 6. **Production Ready**
- Comprehensive API structure
- React frontend
- Monitoring and security
- Complete test suite

This structure supports your requirements for:
- Full multimodal capabilities
- H100 GPU optimization
- SNOMED CT integration
- Natural language patient interface
- Enhanced data scraping
- Kaggle dataset processing
