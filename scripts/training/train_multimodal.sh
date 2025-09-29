#!/bin/bash

# Comprehensive Medical AI Training Pipeline
# Trains multimodal medical AI model with H100 optimization

set -e

# Configuration
CONFIG_DIR="conf"
DATA_DIR="data"
OUTPUT_DIR="outputs"
MODELS_DIR="models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python packages
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
        error "PyTorch not found"
        exit 1
    }
    
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
        error "Transformers not found"
        exit 1
    }
    
    # Check GPU availability
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
    
    # Check H100 GPUs
    if nvidia-smi | grep -q "H100"; then
        log "H100 GPUs detected"
    else
        warning "H100 GPUs not detected - training may be slower"
    fi
}

# Setup environment
setup_environment() {
    log "Setting up training environment..."
    
    # Create output directories
    mkdir -p $OUTPUT_DIR/{logs,checkpoints,metrics,models}
    mkdir -p $MODELS_DIR/{tokenizer,textlm,sft,multimodal}
    
    # Set environment variables for H100 optimization
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TOKENIZERS_PARALLELISM=false
    
    log "Environment setup complete"
}

# Download and prepare data
prepare_data() {
    log "Preparing training data..."
    
    # Download SNOMED CT if not exists
    if [ ! -d "$DATA_DIR/ontologies/snomed" ]; then
        info "Downloading SNOMED CT..."
        python -m src.ontology.snomed.loader --download --extract --load --mapping
    fi
    
    # Download Kaggle datasets
    if [ ! -d "$DATA_DIR/raw/kaggle" ]; then
        info "Downloading Kaggle medical datasets..."
        python -m src.data.scrapers.kaggle.downloader --all
    fi
    
    # Scrape medical literature
    info "Scraping medical literature..."
    python -m src.data.scrapers.medical.pubmed_full \
        --query "medical diagnosis treatment" \
        --max-results 1000 \
        --output-dir "$DATA_DIR/raw/pubmed"
    
    # Process and clean data
    info "Processing and cleaning data..."
    python -m src.data.processing.cleaning.medical_cleaner \
        --input-dir "$DATA_DIR/raw" \
        --output-dir "$DATA_DIR/processed"
    
    log "Data preparation complete"
}

# Train tokenizer
train_tokenizer() {
    log "Training medical tokenizer..."
    
    python -m src.models.tokenizer.medical_tokenizer \
        --corpus-files "$DATA_DIR/processed"/**/*.txt \
        --output-dir "$MODELS_DIR/tokenizer" \
        --vocab-size 200000 \
        --config "$CONFIG_DIR/tokenizer.yaml"
    
    log "Tokenizer training complete"
}

# Train base language model
train_base_model() {
    log "Training base medical language model..."
    
    # Use distributed training for H100
    torchrun --nproc_per_node=8 --master_port=29500 \
        src/models/textlm/training/trainer.py \
        --config "$CONFIG_DIR/train_text.yaml" \
        --output-dir "$OUTPUT_DIR/textlm" \
        --tokenizer-path "$MODELS_DIR/tokenizer" \
        --dataset-path "$DATA_DIR/processed/corpus" \
        --logging-dir "$OUTPUT_DIR/logs/textlm"
    
    log "Base model training complete"
}

# Supervised fine-tuning
train_sft() {
    log "Starting supervised fine-tuning..."
    
    torchrun --nproc_per_node=8 --master_port=29501 \
        src/models/sft/medical_sft.py \
        --config "$CONFIG_DIR/train_sft.yaml" \
        --base-model "$OUTPUT_DIR/textlm/final" \
        --output-dir "$OUTPUT_DIR/sft" \
        --dataset-path "$DATA_DIR/processed/sft" \
        --logging-dir "$OUTPUT_DIR/logs/sft"
    
    log "SFT training complete"
}

# Train multimodal model
train_multimodal() {
    log "Training multimodal medical model..."
    
    torchrun --nproc_per_node=8 --master_port=29502 \
        src/models/multimodal/training/trainer.py \
        --config "$CONFIG_DIR/train_multimodal.yaml" \
        --text-model "$OUTPUT_DIR/sft/final" \
        --output-dir "$OUTPUT_DIR/multimodal" \
        --dataset-path "$DATA_DIR/processed/multimodal" \
        --logging-dir "$OUTPUT_DIR/logs/multimodal"
    
    log "Multimodal training complete"
}

# Evaluate models
evaluate_models() {
    log "Evaluating trained models..."
    
    # Evaluate base model
    python -m src.evaluation.benchmarks.medical_qa_eval \
        --model-path "$OUTPUT_DIR/textlm/final" \
        --output-dir "$OUTPUT_DIR/metrics/textlm"
    
    # Evaluate SFT model
    python -m src.evaluation.benchmarks.medical_qa_eval \
        --model-path "$OUTPUT_DIR/sft/final" \
        --output-dir "$OUTPUT_DIR/metrics/sft"
    
    # Evaluate multimodal model
    python -m src.evaluation.benchmarks.multimodal_eval \
        --model-path "$OUTPUT_DIR/multimodal/final" \
        --output-dir "$OUTPUT_DIR/metrics/multimodal"
    
    # Generate evaluation report
    python -m src.evaluation.visualization.reports \
        --metrics-dir "$OUTPUT_DIR/metrics" \
        --output-file "$OUTPUT_DIR/evaluation_report.html"
    
    log "Evaluation complete"
}

# Deploy models
deploy_models() {
    log "Deploying trained models..."
    
    # Copy final models to deployment directory
    cp -r "$OUTPUT_DIR/multimodal/final" "$MODELS_DIR/multimodal/"
    cp -r "$OUTPUT_DIR/sft/final" "$MODELS_DIR/sft/"
    cp -r "$OUTPUT_DIR/textlm/final" "$MODELS_DIR/textlm/"
    
    # Start API server
    info "Starting API server..."
    python -m src.api.main \
        --config "$CONFIG_DIR/app.yaml" \
        --model-path "$MODELS_DIR/multimodal" \
        --tokenizer-path "$MODELS_DIR/tokenizer" &
    
    # Wait for server to start
    sleep 10
    
    # Test API
    curl -X POST http://localhost:8000/health || {
        error "API server failed to start"
        exit 1
    }
    
    log "Models deployed successfully"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove temporary training files
    find $OUTPUT_DIR -name "*.tmp" -delete
    find $OUTPUT_DIR -name "checkpoint-*" -type d | head -n -3 | xargs rm -rf
    
    # Compress logs
    gzip $OUTPUT_DIR/logs/**/*.log 2>/dev/null || true
    
    log "Cleanup complete"
}

# Main training pipeline
main() {
    log "Starting comprehensive medical AI training pipeline"
    
    # Parse command line arguments
    STAGE="all"
    SKIP_DATA=false
    SKIP_EVAL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stage)
                STAGE="$2"
                shift 2
                ;;
            --skip-data)
                SKIP_DATA=true
                shift
                ;;
            --skip-eval)
                SKIP_EVAL=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --stage STAGE     Run specific stage (data|tokenizer|base|sft|multimodal|eval|deploy|all)"
                echo "  --skip-data       Skip data preparation"
                echo "  --skip-eval       Skip evaluation"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check dependencies
    check_dependencies
    
    # Setup environment
    setup_environment
    
    # Execute pipeline stages
    case $STAGE in
        data)
            prepare_data
            ;;
        tokenizer)
            train_tokenizer
            ;;
        base)
            train_base_model
            ;;
        sft)
            train_sft
            ;;
        multimodal)
            train_multimodal
            ;;
        eval)
            evaluate_models
            ;;
        deploy)
            deploy_models
            ;;
        all)
            if [ "$SKIP_DATA" = false ]; then
                prepare_data
            fi
            
            train_tokenizer
            train_base_model
            train_sft
            train_multimodal
            
            if [ "$SKIP_EVAL" = false ]; then
                evaluate_models
            fi
            
            deploy_models
            cleanup
            ;;
        *)
            error "Unknown stage: $STAGE"
            exit 1
            ;;
    esac
    
    log "Training pipeline completed successfully!"
    
    # Print summary
    echo
    echo "=== TRAINING SUMMARY ==="
    echo "Models saved to: $MODELS_DIR"
    echo "Logs saved to: $OUTPUT_DIR/logs"
    echo "Metrics saved to: $OUTPUT_DIR/metrics"
    echo "API running at: http://localhost:8000"
    echo
    echo "Next steps:"
    echo "1. Review evaluation metrics in $OUTPUT_DIR/evaluation_report.html"
    echo "2. Test the API endpoints"
    echo "3. Deploy to production environment"
    echo
}

# Error handling
trap 'error "Training pipeline failed at line $LINENO"' ERR

# Run main function
main "$@"
