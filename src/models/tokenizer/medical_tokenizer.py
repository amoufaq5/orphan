"""
Medical Domain Tokenizer with H100 Optimization

This module provides a specialized tokenizer for medical text with extensive
medical vocabulary, optimized for H100 GPU training and inference.
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from collections import Counter, defaultdict
import re

import torch
import numpy as np
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    BertTokenizer,
    GPT2Tokenizer
)
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, ByteLevel, Metaspace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer

import sentencepiece as spm
from datasets import Dataset

from ...utils.config.loader import load_config
from ...utils.logging.logger import get_logger
from ...ontology.snomed.loader import SNOMEDLoader
from ...ontology.icd10.mapper import ICD10Mapper
from ...ontology.mesh.mapper import MeSHMapper

logger = get_logger(__name__)


class MedicalVocabularyBuilder:
    """
    Builds comprehensive medical vocabulary from multiple sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path or "conf/tokenizer.yaml")
        self.medical_terms = set()
        self.drug_names = set()
        self.anatomy_terms = set()
        self.procedure_terms = set()
        self.symptom_terms = set()
        
        # Initialize ontology loaders
        self.snomed_loader = SNOMEDLoader()
        self.icd10_mapper = ICD10Mapper()
        self.mesh_mapper = MeSHMapper()
    
    def build_vocabulary(self) -> Dict[str, List[str]]:
        """Build comprehensive medical vocabulary from all sources."""
        logger.info("Building medical vocabulary...")
        
        # Load from ontologies
        self._load_snomed_terms()
        self._load_icd10_terms()
        self._load_mesh_terms()
        
        # Load from medical dictionaries
        self._load_drug_names()
        self._load_anatomy_terms()
        self._load_procedure_terms()
        
        # Load from medical abbreviations
        self._load_medical_abbreviations()
        
        # Compile vocabulary
        vocabulary = {
            "medical_terms": list(self.medical_terms),
            "drug_names": list(self.drug_names),
            "anatomy_terms": list(self.anatomy_terms),
            "procedure_terms": list(self.procedure_terms),
            "symptom_terms": list(self.symptom_terms)
        }
        
        total_terms = sum(len(terms) for terms in vocabulary.values())
        logger.info(f"Built medical vocabulary with {total_terms} unique terms")
        
        return vocabulary
    
    def _load_snomed_terms(self):
        """Load terms from SNOMED CT."""
        try:
            snomed_terms = self.snomed_loader.get_all_concepts()
            for concept in snomed_terms:
                term = concept.get('term', '').lower().strip()
                if term and len(term) > 2:
                    self.medical_terms.add(term)
                    
                    # Categorize by semantic type
                    semantic_type = concept.get('semantic_type', '')
                    if 'drug' in semantic_type or 'medication' in semantic_type:
                        self.drug_names.add(term)
                    elif 'anatomy' in semantic_type or 'body' in semantic_type:
                        self.anatomy_terms.add(term)
                    elif 'procedure' in semantic_type:
                        self.procedure_terms.add(term)
                    elif 'symptom' in semantic_type or 'sign' in semantic_type:
                        self.symptom_terms.add(term)
            
            logger.info(f"Loaded {len(snomed_terms)} SNOMED CT terms")
        except Exception as e:
            logger.warning(f"Failed to load SNOMED terms: {e}")
    
    def _load_icd10_terms(self):
        """Load terms from ICD-10."""
        try:
            icd10_terms = self.icd10_mapper.get_all_codes()
            for code_info in icd10_terms:
                term = code_info.get('description', '').lower().strip()
                if term and len(term) > 2:
                    self.medical_terms.add(term)
                    self.symptom_terms.add(term)  # ICD-10 is mostly diseases/symptoms
            
            logger.info(f"Loaded {len(icd10_terms)} ICD-10 terms")
        except Exception as e:
            logger.warning(f"Failed to load ICD-10 terms: {e}")
    
    def _load_mesh_terms(self):
        """Load terms from MeSH."""
        try:
            mesh_terms = self.mesh_mapper.get_all_descriptors()
            for descriptor in mesh_terms:
                term = descriptor.get('name', '').lower().strip()
                if term and len(term) > 2:
                    self.medical_terms.add(term)
            
            logger.info(f"Loaded {len(mesh_terms)} MeSH terms")
        except Exception as e:
            logger.warning(f"Failed to load MeSH terms: {e}")
    
    def _load_drug_names(self):
        """Load drug names from various sources."""
        # Common drug names (this would be loaded from a comprehensive database)
        common_drugs = [
            "acetaminophen", "ibuprofen", "aspirin", "metformin", "lisinopril",
            "amlodipine", "metoprolol", "omeprazole", "simvastatin", "losartan",
            "hydrochlorothiazide", "atorvastatin", "levothyroxine", "albuterol",
            "furosemide", "prednisone", "tramadol", "gabapentin", "sertraline",
            "citalopram", "escitalopram", "fluoxetine", "paroxetine", "venlafaxine"
        ]
        
        for drug in common_drugs:
            self.drug_names.add(drug.lower())
            self.medical_terms.add(drug.lower())
    
    def _load_anatomy_terms(self):
        """Load anatomical terms."""
        anatomy_terms = [
            "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine",
            "pancreas", "spleen", "gallbladder", "bladder", "uterus", "ovary",
            "prostate", "thyroid", "adrenal", "pituitary", "hypothalamus",
            "cerebrum", "cerebellum", "brainstem", "spinal cord", "vertebra",
            "femur", "tibia", "fibula", "humerus", "radius", "ulna", "scapula"
        ]
        
        for term in anatomy_terms:
            self.anatomy_terms.add(term.lower())
            self.medical_terms.add(term.lower())
    
    def _load_procedure_terms(self):
        """Load medical procedure terms."""
        procedures = [
            "surgery", "biopsy", "endoscopy", "colonoscopy", "bronchoscopy",
            "laparoscopy", "arthroscopy", "angiography", "angioplasty",
            "catheterization", "intubation", "tracheostomy", "thoracentesis",
            "paracentesis", "lumbar puncture", "bone marrow biopsy",
            "appendectomy", "cholecystectomy", "hysterectomy", "mastectomy"
        ]
        
        for procedure in procedures:
            self.procedure_terms.add(procedure.lower())
            self.medical_terms.add(procedure.lower())
    
    def _load_medical_abbreviations(self):
        """Load common medical abbreviations."""
        abbreviations = [
            "bp", "hr", "rr", "temp", "o2", "co2", "ecg", "ekg", "mri", "ct",
            "xray", "ultrasound", "pet", "spect", "eeg", "emg", "cbc", "bmp",
            "cmp", "pt", "ptt", "inr", "bun", "creatinine", "glucose", "hba1c",
            "tsh", "t3", "t4", "psa", "cea", "afp", "ca125", "ca199"
        ]
        
        for abbr in abbreviations:
            self.medical_terms.add(abbr.lower())


class MedicalTokenizer(PreTrainedTokenizerFast):
    """
    Medical domain tokenizer with extensive medical vocabulary and H100 optimizations.
    
    Features:
    - 200K+ medical vocabulary
    - Subword tokenization optimized for medical terms
    - Special tokens for medical entities
    - H100-optimized tensor operations
    - Support for multiple medical ontologies
    """
    
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "tokenizer_file": "tokenizer.json",
        "medical_vocab_file": "medical_vocab.json"
    }
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        medical_vocab_file: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
        max_len: int = 2048,
        **kwargs
    ):
        # Medical special tokens
        additional_special_tokens = [
            "<patient>", "</patient>",
            "<symptom>", "</symptom>",
            "<diagnosis>", "</diagnosis>",
            "<treatment>", "</treatment>",
            "<medication>", "</medication>",
            "<dosage>", "</dosage>",
            "<procedure>", "</procedure>",
            "<anatomy>", "</anatomy>",
            "<lab_result>", "</lab_result>",
            "<vital_sign>", "</vital_sign>",
            "<allergy>", "</allergy>",
            "<contraindication>", "</contraindication>",
            "<red_flag>", "</red_flag>",
            "<wham_who>", "</wham_who>",
            "<wham_how>", "</wham_how>",
            "<wham_action>", "</wham_action>",
            "<wham_monitoring>", "</wham_monitoring>",
            "<as_age>", "</as_age>",
            "<as_selfcare>", "</as_selfcare>",
            "<as_medication>", "</as_medication>",
            "<as_extra>", "</as_extra>",
            "<as_time>", "</as_time>",
            "<as_history>", "</as_history>",
            "<as_othersymptoms>", "</as_othersymptoms>",
            "<as_danger>", "</as_danger>"
        ]
        
        # Load or create tokenizer
        if tokenizer_file and os.path.exists(tokenizer_file):
            tokenizer = Tokenizer.from_file(tokenizer_file)
        else:
            tokenizer = self._create_medical_tokenizer()
        
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            max_len=max_len,
            **kwargs
        )
        
        # Load medical vocabulary
        self.medical_vocab = {}
        if medical_vocab_file and os.path.exists(medical_vocab_file):
            with open(medical_vocab_file, 'r') as f:
                self.medical_vocab = json.load(f)
        
        # H100 optimizations
        self._setup_h100_optimizations()
    
    def _create_medical_tokenizer(self) -> Tokenizer:
        """Create a new medical tokenizer from scratch."""
        logger.info("Creating new medical tokenizer...")
        
        # Use BPE model for better medical term handling
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        
        # Normalization
        tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            Lowercase(),
            StripAccents()
        ])
        
        # Pre-tokenization
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Whitespace(),
            pre_tokenizers.Punctuation()
        ])
        
        # Decoder
        tokenizer.decoder = decoders.BPEDecoder()
        
        return tokenizer
    
    def _setup_h100_optimizations(self):
        """Setup H100-specific optimizations."""
        # Pad to multiples of 8 for tensor core optimization
        self.pad_to_multiple_of = 8
        
        # Enable fast tokenization paths
        self.is_fast = True
        
        # Optimize for batch processing
        self.padding_side = "right"
        self.truncation_side = "right"
    
    def train_from_corpus(
        self,
        corpus_files: List[str],
        vocab_size: int = 200000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """Train tokenizer on medical corpus."""
        logger.info(f"Training medical tokenizer on {len(corpus_files)} files...")
        
        # Build medical vocabulary
        vocab_builder = MedicalVocabularyBuilder()
        medical_vocab = vocab_builder.build_vocabulary()
        
        # Prepare special tokens
        if special_tokens is None:
            special_tokens = [
                "<unk>", "<s>", "</s>", "<pad>", "<mask>"
            ] + self.additional_special_tokens
        
        # Create trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
            continuing_subword_prefix="##"
        )
        
        # Add medical terms to trainer with higher frequency
        medical_terms_file = "temp_medical_terms.txt"
        with open(medical_terms_file, 'w') as f:
            for category, terms in medical_vocab.items():
                for term in terms:
                    # Repeat medical terms to increase their frequency
                    f.write(f"{term}\n" * 10)
        
        # Train tokenizer
        files = corpus_files + [medical_terms_file]
        self._tokenizer.train(files, trainer)
        
        # Clean up temporary file
        os.remove(medical_terms_file)
        
        # Save medical vocabulary
        self.medical_vocab = medical_vocab
        
        logger.info(f"Tokenizer training completed. Vocabulary size: {self.vocab_size}")
    
    def encode_medical_text(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode medical text with optimizations for H100.
        
        Args:
            text: Input medical text
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Truncation strategy
            return_tensors: Return format ("pt" for PyTorch)
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        # Preprocess medical text
        text = self._preprocess_medical_text(text)
        
        # Tokenize
        encoding = self(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length or self.model_max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs
        )
        
        # Add medical entity markers if requested
        if kwargs.get('add_medical_markers', False):
            encoding = self._add_medical_markers(encoding, text)
        
        return encoding
    
    def _preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text for better tokenization."""
        # Normalize medical abbreviations
        text = self._normalize_medical_abbreviations(text)
        
        # Handle medical units
        text = self._normalize_medical_units(text)
        
        # Handle dosages
        text = self._normalize_dosages(text)
        
        return text
    
    def _normalize_medical_abbreviations(self, text: str) -> str:
        """Normalize common medical abbreviations."""
        abbreviations = {
            r'\bbp\b': 'blood pressure',
            r'\bhr\b': 'heart rate',
            r'\brr\b': 'respiratory rate',
            r'\btemp\b': 'temperature',
            r'\bo2\b': 'oxygen',
            r'\bco2\b': 'carbon dioxide',
            r'\becg\b': 'electrocardiogram',
            r'\bekg\b': 'electrocardiogram',
            r'\bmri\b': 'magnetic resonance imaging',
            r'\bct\b': 'computed tomography',
            r'\bcbc\b': 'complete blood count',
            r'\bbmp\b': 'basic metabolic panel',
            r'\bcmp\b': 'comprehensive metabolic panel'
        }
        
        for abbr, full_form in abbreviations.items():
            text = re.sub(abbr, full_form, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_medical_units(self, text: str) -> str:
        """Normalize medical units."""
        # Normalize common units
        unit_patterns = [
            (r'(\d+)\s*mg/dl', r'\1 milligrams per deciliter'),
            (r'(\d+)\s*mmol/l', r'\1 millimoles per liter'),
            (r'(\d+)\s*bpm', r'\1 beats per minute'),
            (r'(\d+)\s*mmhg', r'\1 millimeters of mercury'),
            (r'(\d+)\s*°c', r'\1 degrees celsius'),
            (r'(\d+)\s*°f', r'\1 degrees fahrenheit')
        ]
        
        for pattern, replacement in unit_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_dosages(self, text: str) -> str:
        """Normalize medication dosages."""
        # Normalize dosage patterns
        dosage_patterns = [
            (r'(\d+)\s*mg', r'\1 milligrams'),
            (r'(\d+)\s*g', r'\1 grams'),
            (r'(\d+)\s*ml', r'\1 milliliters'),
            (r'(\d+)\s*mcg', r'\1 micrograms'),
            (r'(\d+)\s*iu', r'\1 international units')
        ]
        
        for pattern, replacement in dosage_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_medical_markers(self, encoding: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Add medical entity markers to encoding."""
        # This would use NER to identify medical entities and add appropriate markers
        # For now, return the encoding as-is
        return encoding
    
    def batch_encode_medical(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch encode medical texts with H100 optimization.
        
        Args:
            texts: List of medical texts
            batch_size: Batch size for processing
            **kwargs: Additional encoding arguments
            
        Returns:
            List of encoded texts
        """
        encodings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_encodings = [
                self.encode_medical_text(text, **kwargs)
                for text in batch_texts
            ]
            encodings.extend(batch_encodings)
        
        return encodings
    
    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Save tokenizer and medical vocabulary."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        super().save_pretrained(save_directory, **kwargs)
        
        # Save medical vocabulary
        medical_vocab_path = save_directory / "medical_vocab.json"
        with open(medical_vocab_path, 'w') as f:
            json.dump(self.medical_vocab, f, indent=2)
        
        logger.info(f"Medical tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        **kwargs
    ) -> "MedicalTokenizer":
        """Load pretrained medical tokenizer."""
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Load medical vocabulary if available
        medical_vocab_path = Path(pretrained_model_name_or_path) / "medical_vocab.json"
        if medical_vocab_path.exists():
            with open(medical_vocab_path, 'r') as f:
                tokenizer.medical_vocab = json.load(f)
        
        return tokenizer
    
    def get_medical_token_ids(self, category: str) -> List[int]:
        """Get token IDs for medical terms in a specific category."""
        if category not in self.medical_vocab:
            return []
        
        token_ids = []
        for term in self.medical_vocab[category]:
            tokens = self.tokenize(term)
            ids = self.convert_tokens_to_ids(tokens)
            token_ids.extend(ids)
        
        return list(set(token_ids))  # Remove duplicates
    
    def is_medical_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a medical term."""
        token = self.convert_ids_to_tokens([token_id])[0]
        token_text = self.convert_tokens_to_string([token]).strip()
        
        # Check against medical vocabulary
        for category, terms in self.medical_vocab.items():
            if token_text.lower() in terms:
                return True
        
        return False


def train_medical_tokenizer(
    corpus_files: List[str],
    output_dir: str,
    vocab_size: int = 200000,
    config_path: Optional[str] = None
) -> MedicalTokenizer:
    """
    Train a medical tokenizer from corpus files.
    
    Args:
        corpus_files: List of corpus file paths
        output_dir: Output directory for trained tokenizer
        vocab_size: Target vocabulary size
        config_path: Path to tokenizer configuration
        
    Returns:
        Trained medical tokenizer
    """
    logger.info(f"Training medical tokenizer with vocab size {vocab_size}")
    
    # Create tokenizer
    tokenizer = MedicalTokenizer()
    
    # Train on corpus
    tokenizer.train_from_corpus(
        corpus_files=corpus_files,
        vocab_size=vocab_size,
        min_frequency=2
    )
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Medical tokenizer training completed and saved to {output_dir}")
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train medical tokenizer")
    parser.add_argument("--corpus_files", nargs="+", required=True, help="Corpus files for training")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=200000, help="Vocabulary size")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    train_medical_tokenizer(
        corpus_files=args.corpus_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        config_path=args.config
    )
