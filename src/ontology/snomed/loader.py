"""
SNOMED CT Loader and Disease-Symptom Mapper

This module provides comprehensive SNOMED CT integration with advanced
disease-symptom mapping capabilities for medical AI applications.
"""

import os
import csv
import json
import pickle
import zipfile
import requests
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import time
from datetime import datetime
import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm
import sqlite3
from sqlalchemy import create_engine, text
import faiss

from ...utils.config.loader import load_config
from ...utils.logging.logger import get_logger
from ...utils.io.file_handler import FileHandler

logger = get_logger(__name__)


@dataclass
class SNOMEDConcept:
    """SNOMED CT Concept representation."""
    id: str
    effective_time: str
    active: bool
    module_id: str
    definition_status_id: str
    fsn: Optional[str] = None  # Fully Specified Name
    preferred_term: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    semantic_type: Optional[str] = None
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)


@dataclass
class SNOMEDRelationship:
    """SNOMED CT Relationship representation."""
    id: str
    effective_time: str
    active: bool
    module_id: str
    source_id: str
    destination_id: str
    relationship_group: int
    type_id: str
    characteristic_type_id: str
    modifier_id: str


@dataclass
class DiseaseSymptomMapping:
    """Disease-Symptom mapping with confidence scores."""
    disease_id: str
    disease_term: str
    symptom_id: str
    symptom_term: str
    relationship_type: str
    confidence: float
    evidence_count: int
    source: str = "SNOMED_CT"


class SNOMEDLoader:
    """
    Comprehensive SNOMED CT loader with disease-symptom mapping capabilities.
    
    Features:
    - Full SNOMED CT RF2 format support
    - Disease-symptom relationship extraction
    - High-performance indexing and search
    - Integration with other medical ontologies
    - Caching and optimization for large datasets
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize SNOMED CT loader."""
        self.config = load_config(config_path or "conf/snomed_config.yaml")
        self.snomed_config = self.config["snomed"]
        self.mapping_config = self.config["disease_symptom_mapping"]
        
        # Setup paths
        self.download_dir = Path(self.snomed_config["paths"]["download_dir"])
        self.extract_dir = Path(self.snomed_config["paths"]["extract_dir"])
        self.processed_dir = Path(self.snomed_config["paths"]["processed_dir"])
        self.index_dir = Path(self.snomed_config["paths"]["index_dir"])
        
        # Create directories
        for path in [self.download_dir, self.extract_dir, self.processed_dir, self.index_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data structures
        self.concepts: Dict[str, SNOMEDConcept] = {}
        self.relationships: Dict[str, SNOMEDRelationship] = {}
        self.descriptions: Dict[str, Dict] = {}
        self.disease_symptom_map: Dict[str, List[DiseaseSymptomMapping]] = defaultdict(list)
        
        # Initialize database connection
        self.db_path = self.processed_dir / "snomed_ct.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        
        # Initialize search index
        self.search_index = None
        self.concept_embeddings = {}
        
        logger.info("SNOMED CT Loader initialized")
    
    def download_snomed_ct(self, force_download: bool = False) -> bool:
        """
        Download SNOMED CT release files.
        
        Args:
            force_download: Force re-download even if files exist
            
        Returns:
            True if download successful, False otherwise
        """
        download_url = self.snomed_config["download"]["base_url"]
        zip_filename = download_url.split("/")[-1]
        zip_path = self.download_dir / zip_filename
        
        if zip_path.exists() and not force_download:
            logger.info(f"SNOMED CT file already exists: {zip_path}")
            return True
        
        logger.info(f"Downloading SNOMED CT from {download_url}")
        
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading SNOMED CT",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"SNOMED CT downloaded successfully to {zip_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download SNOMED CT: {e}")
            return False
    
    def extract_snomed_ct(self) -> bool:
        """
        Extract SNOMED CT files from downloaded archive.
        
        Returns:
            True if extraction successful, False otherwise
        """
        zip_files = list(self.download_dir.glob("*.zip"))
        if not zip_files:
            logger.error("No SNOMED CT zip files found")
            return False
        
        zip_path = zip_files[0]  # Use the first zip file found
        
        logger.info(f"Extracting SNOMED CT from {zip_path}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            
            logger.info(f"SNOMED CT extracted to {self.extract_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract SNOMED CT: {e}")
            return False
    
    def load_concepts(self) -> bool:
        """Load SNOMED CT concepts from RF2 files."""
        logger.info("Loading SNOMED CT concepts...")
        
        # Find concepts file
        concept_pattern = self.snomed_config["core_files"]["concepts"]
        concept_files = list(self.extract_dir.rglob(concept_pattern.replace("*", "*")))
        
        if not concept_files:
            logger.error(f"No concept files found matching pattern: {concept_pattern}")
            return False
        
        concept_file = concept_files[0]
        logger.info(f"Loading concepts from {concept_file}")
        
        try:
            # Load concepts in chunks for memory efficiency
            chunk_size = self.config["processing"]["batch_size"]["concepts"]
            
            for chunk in pd.read_csv(concept_file, sep='\t', chunksize=chunk_size, dtype=str):
                for _, row in chunk.iterrows():
                    if row['active'] == '1':  # Only active concepts
                        concept = SNOMEDConcept(
                            id=row['id'],
                            effective_time=row['effectiveTime'],
                            active=True,
                            module_id=row['moduleId'],
                            definition_status_id=row['definitionStatusId']
                        )
                        self.concepts[concept.id] = concept
            
            logger.info(f"Loaded {len(self.concepts)} active concepts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load concepts: {e}")
            return False
    
    def load_descriptions(self) -> bool:
        """Load SNOMED CT descriptions (terms) from RF2 files."""
        logger.info("Loading SNOMED CT descriptions...")
        
        # Find descriptions file
        desc_pattern = self.snomed_config["core_files"]["descriptions"]
        desc_files = list(self.extract_dir.rglob(desc_pattern.replace("*", "*")))
        
        if not desc_files:
            logger.error(f"No description files found matching pattern: {desc_pattern}")
            return False
        
        desc_file = desc_files[0]
        logger.info(f"Loading descriptions from {desc_file}")
        
        try:
            # Description type IDs
            FSN_TYPE = "900000000000003001"  # Fully Specified Name
            SYNONYM_TYPE = "900000000000013009"  # Synonym
            
            chunk_size = self.config["processing"]["batch_size"]["descriptions"]
            
            for chunk in pd.read_csv(desc_file, sep='\t', chunksize=chunk_size, dtype=str):
                for _, row in chunk.iterrows():
                    if row['active'] == '1' and row['languageCode'] == 'en':
                        concept_id = row['conceptId']
                        
                        if concept_id in self.concepts:
                            concept = self.concepts[concept_id]
                            
                            if row['typeId'] == FSN_TYPE:
                                concept.fsn = row['term']
                            elif row['typeId'] == SYNONYM_TYPE:
                                if not concept.preferred_term:
                                    concept.preferred_term = row['term']
                                concept.synonyms.append(row['term'])
            
            logger.info("Descriptions loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load descriptions: {e}")
            return False
    
    def load_relationships(self) -> bool:
        """Load SNOMED CT relationships from RF2 files."""
        logger.info("Loading SNOMED CT relationships...")
        
        # Find relationships file
        rel_pattern = self.snomed_config["core_files"]["relationships"]
        rel_files = list(self.extract_dir.rglob(rel_pattern.replace("*", "*")))
        
        if not rel_files:
            logger.error(f"No relationship files found matching pattern: {rel_pattern}")
            return False
        
        rel_file = rel_files[0]
        logger.info(f"Loading relationships from {rel_file}")
        
        try:
            chunk_size = self.config["processing"]["batch_size"]["relationships"]
            
            for chunk in pd.read_csv(rel_file, sep='\t', chunksize=chunk_size, dtype=str):
                for _, row in chunk.iterrows():
                    if row['active'] == '1':  # Only active relationships
                        relationship = SNOMEDRelationship(
                            id=row['id'],
                            effective_time=row['effectiveTime'],
                            active=True,
                            module_id=row['moduleId'],
                            source_id=row['sourceId'],
                            destination_id=row['destinationId'],
                            relationship_group=int(row['relationshipGroup']),
                            type_id=row['typeId'],
                            characteristic_type_id=row['characteristicTypeId'],
                            modifier_id=row['modifierId']
                        )
                        self.relationships[relationship.id] = relationship
                        
                        # Build parent-child relationships
                        if row['typeId'] == "116680003":  # "Is a" relationship
                            source_concept = self.concepts.get(row['sourceId'])
                            dest_concept = self.concepts.get(row['destinationId'])
                            
                            if source_concept and dest_concept:
                                source_concept.parent_concepts.append(row['destinationId'])
                                dest_concept.child_concepts.append(row['sourceId'])
            
            logger.info(f"Loaded {len(self.relationships)} active relationships")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load relationships: {e}")
            return False
    
    def build_disease_symptom_mapping(self) -> bool:
        """
        Build comprehensive disease-symptom mapping from SNOMED CT relationships.
        
        Returns:
            True if mapping built successfully, False otherwise
        """
        logger.info("Building disease-symptom mapping...")
        
        # Define semantic types for diseases and symptoms
        disease_types = {
            "64572001",    # Disease
            "404684003",   # Clinical finding
            "272379006",   # Event
            "118940003",   # Disorder of anatomical structure
            "362965005",   # Disorder of body system
        }
        
        symptom_types = {
            "418799008",   # Finding reported by subject or history provider
            "404684003",   # Clinical finding
            "72670004",    # Sign
            "162408000",   # Symptom
        }
        
        # Relationship types that indicate symptom associations
        symptom_relationships = {
            "47429007",    # Associated with
            "363714003",   # Interprets
            "246090004",   # Associated finding
            "116676008",   # Associated morphology
            "363698007",   # Finding site
        }
        
        mapping_count = 0
        
        try:
            # Process relationships to find disease-symptom associations
            for rel_id, relationship in tqdm(self.relationships.items(), desc="Processing relationships"):
                if relationship.type_id in symptom_relationships:
                    source_concept = self.concepts.get(relationship.source_id)
                    dest_concept = self.concepts.get(relationship.destination_id)
                    
                    if source_concept and dest_concept:
                        # Determine which is disease and which is symptom
                        disease_concept = None
                        symptom_concept = None
                        
                        # Check if source is disease and destination is symptom
                        if (self._is_concept_type(source_concept, disease_types) and 
                            self._is_concept_type(dest_concept, symptom_types)):
                            disease_concept = source_concept
                            symptom_concept = dest_concept
                        
                        # Check if source is symptom and destination is disease
                        elif (self._is_concept_type(source_concept, symptom_types) and 
                              self._is_concept_type(dest_concept, disease_types)):
                            disease_concept = dest_concept
                            symptom_concept = source_concept
                        
                        if disease_concept and symptom_concept:
                            # Calculate confidence based on relationship type and evidence
                            confidence = self._calculate_mapping_confidence(
                                relationship.type_id,
                                disease_concept,
                                symptom_concept
                            )
                            
                            mapping = DiseaseSymptomMapping(
                                disease_id=disease_concept.id,
                                disease_term=disease_concept.preferred_term or disease_concept.fsn or "Unknown",
                                symptom_id=symptom_concept.id,
                                symptom_term=symptom_concept.preferred_term or symptom_concept.fsn or "Unknown",
                                relationship_type=relationship.type_id,
                                confidence=confidence,
                                evidence_count=1,
                                source="SNOMED_CT"
                            )
                            
                            self.disease_symptom_map[disease_concept.id].append(mapping)
                            mapping_count += 1
            
            # Consolidate mappings and calculate final confidence scores
            self._consolidate_mappings()
            
            logger.info(f"Built disease-symptom mapping with {mapping_count} associations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build disease-symptom mapping: {e}")
            return False
    
    def _is_concept_type(self, concept: SNOMEDConcept, type_ids: Set[str]) -> bool:
        """Check if a concept belongs to specific semantic types."""
        # Check direct type
        if concept.id in type_ids:
            return True
        
        # Check parent concepts (traverse hierarchy)
        for parent_id in concept.parent_concepts:
            if parent_id in type_ids:
                return True
            
            # Recursive check (limited depth to avoid infinite loops)
            parent_concept = self.concepts.get(parent_id)
            if parent_concept and len(parent_concept.parent_concepts) < 10:
                if self._is_concept_type(parent_concept, type_ids):
                    return True
        
        return False
    
    def _calculate_mapping_confidence(
        self,
        relationship_type: str,
        disease_concept: SNOMEDConcept,
        symptom_concept: SNOMEDConcept
    ) -> float:
        """Calculate confidence score for disease-symptom mapping."""
        base_confidence = 0.5
        
        # Relationship type confidence
        relationship_weights = {
            "47429007": 0.9,   # Associated with
            "363714003": 0.8,  # Interprets
            "246090004": 0.85, # Associated finding
            "116676008": 0.7,  # Associated morphology
            "363698007": 0.75, # Finding site
        }
        
        confidence = relationship_weights.get(relationship_type, base_confidence)
        
        # Adjust based on concept specificity
        if disease_concept.definition_status_id == "900000000000074008":  # Primitive
            confidence *= 0.9
        
        if symptom_concept.definition_status_id == "900000000000074008":  # Primitive
            confidence *= 0.9
        
        return min(confidence, 1.0)
    
    def _consolidate_mappings(self):
        """Consolidate duplicate mappings and update confidence scores."""
        logger.info("Consolidating disease-symptom mappings...")
        
        for disease_id, mappings in self.disease_symptom_map.items():
            # Group mappings by symptom
            symptom_groups = defaultdict(list)
            for mapping in mappings:
                symptom_groups[mapping.symptom_id].append(mapping)
            
            # Consolidate each symptom group
            consolidated_mappings = []
            for symptom_id, symptom_mappings in symptom_groups.items():
                if len(symptom_mappings) == 1:
                    consolidated_mappings.append(symptom_mappings[0])
                else:
                    # Merge multiple mappings for same symptom
                    best_mapping = max(symptom_mappings, key=lambda x: x.confidence)
                    best_mapping.evidence_count = len(symptom_mappings)
                    
                    # Boost confidence based on evidence count
                    evidence_boost = min(0.1 * (len(symptom_mappings) - 1), 0.3)
                    best_mapping.confidence = min(best_mapping.confidence + evidence_boost, 1.0)
                    
                    consolidated_mappings.append(best_mapping)
            
            self.disease_symptom_map[disease_id] = consolidated_mappings
    
    def save_disease_symptom_mapping(self) -> bool:
        """Save disease-symptom mapping to various formats."""
        logger.info("Saving disease-symptom mapping...")
        
        try:
            # Prepare data for export
            all_mappings = []
            for disease_id, mappings in self.disease_symptom_map.items():
                for mapping in mappings:
                    all_mappings.append({
                        'disease_id': mapping.disease_id,
                        'disease_term': mapping.disease_term,
                        'symptom_id': mapping.symptom_id,
                        'symptom_term': mapping.symptom_term,
                        'relationship_type': mapping.relationship_type,
                        'confidence': mapping.confidence,
                        'evidence_count': mapping.evidence_count,
                        'source': mapping.source
                    })
            
            # Save as JSON
            json_path = Path(self.mapping_config["output"]["json_file"])
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(all_mappings, f, indent=2)
            
            # Save as CSV
            csv_path = Path(self.mapping_config["output"]["csv_file"])
            df = pd.DataFrame(all_mappings)
            df.to_csv(csv_path, index=False)
            
            # Save as Pickle
            pickle_path = Path(self.mapping_config["output"]["pickle_file"])
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.disease_symptom_map, f)
            
            logger.info(f"Disease-symptom mapping saved to {json_path}, {csv_path}, {pickle_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save disease-symptom mapping: {e}")
            return False
    
    def search_concepts(
        self,
        query: str,
        limit: int = 10,
        semantic_types: Optional[List[str]] = None
    ) -> List[SNOMEDConcept]:
        """
        Search SNOMED CT concepts by text query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            semantic_types: Filter by semantic types
            
        Returns:
            List of matching concepts
        """
        query_lower = query.lower()
        matches = []
        
        for concept in self.concepts.values():
            # Check preferred term
            if concept.preferred_term and query_lower in concept.preferred_term.lower():
                matches.append((concept, 1.0))  # Exact match in preferred term
                continue
            
            # Check FSN
            if concept.fsn and query_lower in concept.fsn.lower():
                matches.append((concept, 0.9))  # Match in FSN
                continue
            
            # Check synonyms
            for synonym in concept.synonyms:
                if query_lower in synonym.lower():
                    matches.append((concept, 0.8))  # Match in synonym
                    break
        
        # Sort by relevance score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by semantic types if specified
        if semantic_types:
            filtered_matches = []
            for concept, score in matches:
                if any(self._is_concept_type(concept, {st}) for st in semantic_types):
                    filtered_matches.append((concept, score))
            matches = filtered_matches
        
        return [concept for concept, _ in matches[:limit]]
    
    def get_disease_symptoms(
        self,
        disease_id: str,
        min_confidence: float = 0.5
    ) -> List[DiseaseSymptomMapping]:
        """
        Get symptoms associated with a disease.
        
        Args:
            disease_id: SNOMED CT disease concept ID
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of associated symptoms
        """
        mappings = self.disease_symptom_map.get(disease_id, [])
        return [m for m in mappings if m.confidence >= min_confidence]
    
    def get_symptom_diseases(
        self,
        symptom_id: str,
        min_confidence: float = 0.5
    ) -> List[DiseaseSymptomMapping]:
        """
        Get diseases associated with a symptom.
        
        Args:
            symptom_id: SNOMED CT symptom concept ID
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of associated diseases
        """
        associated_diseases = []
        
        for disease_id, mappings in self.disease_symptom_map.items():
            for mapping in mappings:
                if mapping.symptom_id == symptom_id and mapping.confidence >= min_confidence:
                    associated_diseases.append(mapping)
        
        return associated_diseases
    
    def load_full_snomed_ct(self) -> bool:
        """
        Load complete SNOMED CT dataset.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Loading full SNOMED CT dataset...")
        
        # Download if needed
        if not self.download_snomed_ct():
            return False
        
        # Extract if needed
        if not self.extract_snomed_ct():
            return False
        
        # Load core components
        if not self.load_concepts():
            return False
        
        if not self.load_descriptions():
            return False
        
        if not self.load_relationships():
            return False
        
        # Build disease-symptom mapping
        if not self.build_disease_symptom_mapping():
            return False
        
        # Save mapping
        if not self.save_disease_symptom_mapping():
            return False
        
        logger.info("SNOMED CT loading completed successfully")
        return True
    
    def get_all_concepts(self) -> List[Dict[str, Any]]:
        """Get all loaded concepts as dictionaries."""
        concepts_list = []
        for concept in self.concepts.values():
            concepts_list.append({
                'id': concept.id,
                'term': concept.preferred_term or concept.fsn,
                'fsn': concept.fsn,
                'synonyms': concept.synonyms,
                'semantic_type': concept.semantic_type,
                'active': concept.active
            })
        return concepts_list


def main():
    """Main function for SNOMED CT loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load SNOMED CT and build disease-symptom mapping")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--download", action="store_true", help="Force download SNOMED CT")
    parser.add_argument("--extract", action="store_true", help="Force extract SNOMED CT")
    parser.add_argument("--load", action="store_true", help="Load SNOMED CT data")
    parser.add_argument("--mapping", action="store_true", help="Build disease-symptom mapping")
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = SNOMEDLoader(config_path=args.config)
    
    # Execute requested operations
    if args.download:
        loader.download_snomed_ct(force_download=True)
    
    if args.extract:
        loader.extract_snomed_ct()
    
    if args.load:
        loader.load_concepts()
        loader.load_descriptions()
        loader.load_relationships()
    
    if args.mapping:
        loader.build_disease_symptom_mapping()
        loader.save_disease_symptom_mapping()
    
    # If no specific operation requested, do full load
    if not any([args.download, args.extract, args.load, args.mapping]):
        loader.load_full_snomed_ct()


if __name__ == "__main__":
    main()
