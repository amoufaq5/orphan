# Orphan Medical AI Platform

> **Comprehensive Multimodal Medical AI Platform** — Advanced healthcare AI system with NHS integration, WHAM/AS METHOD protocols, and H100 GPU optimization.

## Project Overview

**Orphan** is a production-ready medical AI platform designed for UK healthcare integration, featuring:

- **Custom Medical LLM**: 7B+ parameter transformer optimized for medical domain
- **Multimodal Capabilities**: Text + medical imaging (X-rays, CT, MRI, ultrasound)
- **NHS Integration**: UK healthcare system compatibility with SNOMED CT
- **H100 Optimization**: Advanced GPU training and inference optimization
- **Medical Safety**: WHAM/AS METHOD protocols with red flag detection
- **Comprehensive Data**: Full medical literature, not just abstracts

## Target Markets

- **Primary**: UK NHS patients and healthcare providers
- **Secondary**: Medical students and researchers
- **Tertiary**: Healthcare professionals globally

## Architecture

### Core Components

- **Medical Language Model**: Custom transformer with 200K medical vocabulary
- **Vision Encoder**: Specialized medical image analysis
- **SNOMED CT Integration**: Disease-symptom mapping and clinical coding
- **Patient Interface**: Natural language consultation with safety protocols
- **Data Pipeline**: Enhanced scrapers for complete medical literature
- **Training Infrastructure**: H100-optimized distributed training

### Technology Stack

- **Frontend**: React + TypeScript
- **Backend**: Python FastAPI + Node.js
- **Database**: PostgreSQL + ChromaDB
- **AI/ML**: PyTorch + Transformers + Custom architectures
- **Infrastructure**: AWS/GCP hybrid with H100 GPUs

## Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 32GB+ RAM (64GB+ recommended)
- H100 GPUs (optional but recommended)
```

### Installation

```bash
# Clone repository
git clone https://github.com/amoufaq5/orphan.git
cd orphan

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Setup directories
mkdir -p data/{raw,processed,corpus,ontologies,index}
mkdir -p models/{tokenizer,textlm,sft,multimodal}
mkdir -p outputs/{logs,checkpoints,metrics}
```

### Configuration

```bash
# Configure for your environment
cp conf/app.yaml.example conf/app.yaml
cp conf/h100_config.yaml.example conf/h100_config.yaml

# Edit configuration files as needed
vim conf/app.yaml
vim conf/snomed_config.yaml
```

### Training Pipeline

```bash
# Full training pipeline (requires H100 GPUs)
./scripts/training/train_multimodal.sh

# Or run individual stages
./scripts/training/train_multimodal.sh --stage tokenizer
./scripts/training/train_multimodal.sh --stage base
./scripts/training/train_multimodal.sh --stage sft
./scripts/training/train_multimodal.sh --stage multimodal
```

### API Server

```bash
# Start the API server
python -m src.api.main --config conf/app.yaml

# Test the API
curl -X POST http://localhost:8000/health
```

### Patient Interface

```bash
# Start the Gradio interface
python gradio_app.py

# Or use the CLI interface
python -m src.chat.patient_cli
```

## Medical Protocols

### WHAM Protocol
- **W**ho: Patient demographics and context
- **H**ow: Symptom presentation and characteristics
- **A**ction: Treatment recommendations
- **M**onitoring: Follow-up and safety checks

### AS METHOD Protocol
- **A**ge: Age-specific considerations
- **S**elf-care: Self-treatment attempts
- **M**edication: Current medications
- **E**xtra info: Additional patient information
- **T**ime: Timeline of symptoms
- **H**istory: Medical history
- **O**ther symptoms: Associated symptoms
- **D**anger: Red flag detection

## Safety Features

- **Red Flag Detection**: Automatic identification of emergency symptoms
- **Clinical Decision Support**: Evidence-based recommendations
- **Safety Netting**: Clear guidance on when to seek medical attention
- **Contraindication Checking**: Drug interaction and allergy warnings
- **Audit Logging**: Complete interaction tracking for safety

## Performance Metrics

### Target Metrics (3 months)
- **1,000+** UK patients served
- **10,000+** symptom assessments completed
- **100%** red flag detection accuracy
- **90%+** patient satisfaction
- **10M+** medical documents processed

### Model Performance
- **Medical QA Accuracy**: 85%+ on USMLE-style questions
- **Symptom Recognition**: 92%+ accuracy on clinical presentations
- **Safety Detection**: 99%+ sensitivity for red flags
- **Response Time**: <2 seconds for text, <5 seconds for multimodal

## Project Structure

```
orphan/
├── src/                    # Source code
│   ├── api/               # FastAPI backend
│   ├── models/            # AI/ML models
│   ├── data/              # Data processing
│   ├── ontology/          # Medical ontologies (SNOMED CT)
│   ├── protocols/         # Medical protocols (WHAM/AS METHOD)
│   ├── chat/              # Patient interface
│   └── utils/             # Utilities
├── conf/                  # Configuration files
├── scripts/               # Operational scripts
├── data/                  # Data storage
├── models/                # Trained models
├── outputs/               # Training outputs
└── docs/                  # Documentation
```

## Development

### Data Sources

**UK Sources:**
- NHS Digital
- NICE Guidelines
- BMJ Articles
- Cochrane Reviews
- RCGP Resources
- BNF Database

**Global Sources:**
- PubMed Central (full articles)
- ClinicalTrials.gov
- WHO Guidelines
- FDA Resources
- Medical journals

### Model Training

```bash
# Download medical datasets
python -m src.data.scrapers.kaggle.downloader --all

# Process SNOMED CT
python -m src.ontology.snomed.loader --download --extract --load --mapping

# Train custom tokenizer
python -m src.models.tokenizer.medical_tokenizer \
    --corpus-files data/processed/**/*.txt \
    --vocab-size 200000

# Train base model (requires H100)
torchrun --nproc_per_node=8 \
    src/models/textlm/training/trainer.py \
    --config conf/train_text.yaml
```

### Testing

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run end-to-end tests
python -m pytest tests/e2e/

# Test patient interface
python -m src.chat.patient_interface --test-mode
```

## Deployment

### Local Development
```bash
# Start all services
docker-compose up -d

# Or start individually
python -m src.api.main &
python gradio_app.py &
```

### Production Deployment
```bash
# Build Docker images
docker build -t orphan-api .
docker build -t orphan-frontend ./src/frontend

# Deploy to cloud
./scripts/deployment/deploy_production.sh
```

## Documentation

- **[API Documentation](docs/api/)**: Complete API reference
- **[Model Documentation](docs/models/)**: Model architectures and training
- **[Medical Protocols](docs/medical/)**: WHAM/AS METHOD implementation
- **[Deployment Guide](docs/deployment/)**: Production deployment instructions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Medical Disclaimer

This software is for informational purposes only and is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Support

- **Issues**: [GitHub Issues](https://github.com/amoufaq5/orphan/issues)
- **Discussions**: [GitHub Discussions](https://github.com/amoufaq5/orphan/discussions)
- **Email**: support@orphan-medical.ai

## Acknowledgments

- NHS Digital for healthcare data standards
- SNOMED International for clinical terminology
- The medical AI research community
- Open source contributors

---

**Built with ❤️ for better healthcare outcomes**
