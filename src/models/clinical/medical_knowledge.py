"""
General Medical Knowledge System

Comprehensive medical information retrieval and question answering
system for general medical queries, health education, and clinical guidance.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from ...ontology.snomed.loader import SNOMEDLoader
from ...rag.medical_rag import MedicalRAG
from ...utils.logging.logger import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of medical queries."""
    CONDITION_INFO = "condition_info"
    SYMPTOM_INFO = "symptom_info"
    TREATMENT_INFO = "treatment_info"
    MEDICATION_INFO = "medication_info"
    PREVENTION = "prevention"
    ANATOMY = "anatomy"
    PHYSIOLOGY = "physiology"
    DIAGNOSTIC_TEST = "diagnostic_test"
    LIFESTYLE = "lifestyle"
    GENERAL_HEALTH = "general_health"


@dataclass
class MedicalKnowledgeResponse:
    """Response from medical knowledge system."""
    query: str
    query_type: QueryType
    answer: str
    sources: List[Dict[str, str]]
    related_topics: List[str]
    confidence: float
    medical_disclaimer: str
    snomed_codes: List[str]
    additional_resources: List[Dict[str, str]]


class MedicalKnowledgeSystem:
    """
    Comprehensive medical knowledge and information system.
    
    Features:
    - Medical condition information
    - Symptom explanations
    - Treatment options
    - Medication information
    - Prevention strategies
    - Anatomy and physiology
    - Diagnostic test explanations
    - Health education content
    - Evidence-based responses
    - SNOMED CT integration
    """
    
    def __init__(self, knowledge_base_path: str, snomed_path: str, rag_system_path: str):
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.snomed_loader = SNOMEDLoader(snomed_path)
        self.rag_system = MedicalRAG(rag_system_path)
        
        # Medical disclaimer
        self.medical_disclaimer = (
            "This information is for educational purposes only and should not replace "
            "professional medical advice. Always consult with a qualified healthcare "
            "provider for medical concerns, diagnosis, or treatment decisions."
        )
        
        logger.info("Medical Knowledge System initialized")
    
    def answer_medical_query(self, query: str, context: Optional[Dict] = None) -> MedicalKnowledgeResponse:
        """
        Answer general medical questions with comprehensive information.
        
        Args:
            query: Medical question or topic
            context: Additional context (patient demographics, etc.)
            
        Returns:
            Comprehensive medical knowledge response
        """
        # Classify query type
        query_type = self._classify_query(query)
        
        # Extract medical entities
        medical_entities = self._extract_medical_entities(query)
        
        # Retrieve relevant information
        knowledge_response = self._retrieve_medical_knowledge(query, query_type, medical_entities)
        
        # Generate comprehensive answer
        answer = self._generate_comprehensive_answer(
            query, query_type, knowledge_response, context
        )
        
        # Find related topics
        related_topics = self._find_related_topics(medical_entities, query_type)
        
        # Map to SNOMED codes
        snomed_codes = self._map_to_snomed_codes(medical_entities)
        
        # Get additional resources
        additional_resources = self._get_additional_resources(query_type, medical_entities)
        
        return MedicalKnowledgeResponse(
            query=query,
            query_type=query_type,
            answer=answer,
            sources=knowledge_response.get("sources", []),
            related_topics=related_topics,
            confidence=knowledge_response.get("confidence", 0.8),
            medical_disclaimer=self.medical_disclaimer,
            snomed_codes=snomed_codes,
            additional_resources=additional_resources
        )
    
    def get_condition_information(self, condition: str) -> Dict[str, Any]:
        """Get comprehensive information about a medical condition."""
        # Search knowledge base and RAG system
        condition_info = {
            "name": condition,
            "definition": "",
            "symptoms": [],
            "causes": [],
            "risk_factors": [],
            "diagnosis": [],
            "treatment_options": [],
            "prognosis": "",
            "prevention": [],
            "complications": [],
            "when_to_see_doctor": []
        }
        
        # Use RAG system to retrieve detailed information
        rag_response = self.rag_system.query(
            f"comprehensive information about {condition} including symptoms, causes, treatment, prognosis"
        )
        
        # Parse and structure the response
        condition_info = self._parse_condition_information(rag_response, condition)
        
        return condition_info
    
    def get_symptom_information(self, symptom: str) -> Dict[str, Any]:
        """Get information about a specific symptom."""
        symptom_info = {
            "symptom": symptom,
            "description": "",
            "possible_causes": [],
            "associated_symptoms": [],
            "when_to_worry": [],
            "home_remedies": [],
            "when_to_see_doctor": [],
            "diagnostic_tests": []
        }
        
        # Retrieve symptom information
        rag_response = self.rag_system.query(
            f"detailed information about {symptom} symptom causes treatment when to see doctor"
        )
        
        symptom_info = self._parse_symptom_information(rag_response, symptom)
        
        return symptom_info
    
    def get_medication_information(self, medication: str) -> Dict[str, Any]:
        """Get comprehensive medication information."""
        med_info = {
            "name": medication,
            "generic_name": "",
            "drug_class": "",
            "uses": [],
            "how_it_works": "",
            "dosage": "",
            "side_effects": {
                "common": [],
                "serious": [],
                "rare": []
            },
            "contraindications": [],
            "interactions": [],
            "pregnancy_category": "",
            "storage": "",
            "missed_dose": "",
            "overdose": ""
        }
        
        # Retrieve medication information
        rag_response = self.rag_system.query(
            f"comprehensive information about {medication} medication uses dosage side effects interactions"
        )
        
        med_info = self._parse_medication_information(rag_response, medication)
        
        return med_info
    
    def get_diagnostic_test_information(self, test: str) -> Dict[str, Any]:
        """Get information about diagnostic tests."""
        test_info = {
            "test_name": test,
            "purpose": "",
            "preparation": [],
            "procedure": "",
            "duration": "",
            "risks": [],
            "results": {
                "normal_values": "",
                "abnormal_values": "",
                "interpretation": ""
            },
            "follow_up": []
        }
        
        # Retrieve test information
        rag_response = self.rag_system.query(
            f"information about {test} diagnostic test procedure preparation results interpretation"
        )
        
        test_info = self._parse_test_information(rag_response, test)
        
        return test_info
    
    def get_prevention_advice(self, condition: str) -> Dict[str, Any]:
        """Get prevention strategies for medical conditions."""
        prevention_info = {
            "condition": condition,
            "primary_prevention": [],  # Prevent disease occurrence
            "secondary_prevention": [],  # Early detection
            "tertiary_prevention": [],  # Prevent complications
            "lifestyle_modifications": [],
            "screening_recommendations": [],
            "vaccination": [],
            "risk_reduction": []
        }
        
        # Retrieve prevention information
        rag_response = self.rag_system.query(
            f"prevention strategies for {condition} lifestyle modifications screening vaccination"
        )
        
        prevention_info = self._parse_prevention_information(rag_response, condition)
        
        return prevention_info
    
    def _load_knowledge_base(self, knowledge_base_path: str) -> Dict[str, Any]:
        """Load structured medical knowledge base."""
        # This would load processed medical knowledge from various sources
        knowledge_base = {
            "conditions": {},
            "symptoms": {},
            "medications": {},
            "procedures": {},
            "anatomy": {},
            "physiology": {}
        }
        
        # Load from processed medical datasets
        try:
            with open(knowledge_base_path, 'r') as f:
                knowledge_base = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {knowledge_base_path}")
        
        return knowledge_base
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of medical query."""
        query_lower = query.lower()
        
        # Condition information patterns
        condition_patterns = [
            r"what is", r"tell me about", r"information about", r"explain",
            r"symptoms of", r"causes of", r"treatment for"
        ]
        
        # Symptom information patterns
        symptom_patterns = [
            r"why do i have", r"what causes", r"is it normal to",
            r"should i worry about", r"home remedies for"
        ]
        
        # Medication patterns
        medication_patterns = [
            r"side effects of", r"how does.*work", r"dosage of",
            r"interactions with", r"can i take"
        ]
        
        # Prevention patterns
        prevention_patterns = [
            r"how to prevent", r"avoid", r"reduce risk of",
            r"prevention of", r"protect against"
        ]
        
        # Test patterns
        test_patterns = [
            r"what is.*test", r"how is.*diagnosed", r"preparation for",
            r"results of", r"normal values"
        ]
        
        # Check patterns
        if any(re.search(pattern, query_lower) for pattern in condition_patterns):
            return QueryType.CONDITION_INFO
        elif any(re.search(pattern, query_lower) for pattern in symptom_patterns):
            return QueryType.SYMPTOM_INFO
        elif any(re.search(pattern, query_lower) for pattern in medication_patterns):
            return QueryType.MEDICATION_INFO
        elif any(re.search(pattern, query_lower) for pattern in prevention_patterns):
            return QueryType.PREVENTION
        elif any(re.search(pattern, query_lower) for pattern in test_patterns):
            return QueryType.DIAGNOSTIC_TEST
        else:
            return QueryType.GENERAL_HEALTH
    
    def _extract_medical_entities(self, query: str) -> List[str]:
        """Extract medical entities from query using SNOMED CT."""
        entities = []
        
        # Use SNOMED CT to identify medical terms
        snomed_matches = self.snomed_loader.search_concepts(query, limit=10)
        
        for match in snomed_matches:
            if match["confidence"] > 0.7:
                entities.append(match["preferred_term"])
        
        return entities
    
    def _retrieve_medical_knowledge(
        self,
        query: str,
        query_type: QueryType,
        entities: List[str]
    ) -> Dict[str, Any]:
        """Retrieve relevant medical knowledge."""
        # Use RAG system for comprehensive retrieval
        rag_response = self.rag_system.query(query)
        
        # Enhance with structured knowledge
        structured_info = {}
        
        for entity in entities:
            if query_type == QueryType.CONDITION_INFO:
                structured_info[entity] = self.knowledge_base.get("conditions", {}).get(entity, {})
            elif query_type == QueryType.SYMPTOM_INFO:
                structured_info[entity] = self.knowledge_base.get("symptoms", {}).get(entity, {})
            elif query_type == QueryType.MEDICATION_INFO:
                structured_info[entity] = self.knowledge_base.get("medications", {}).get(entity, {})
        
        return {
            "rag_response": rag_response,
            "structured_info": structured_info,
            "sources": rag_response.get("sources", []),
            "confidence": rag_response.get("confidence", 0.8)
        }
    
    def _generate_comprehensive_answer(
        self,
        query: str,
        query_type: QueryType,
        knowledge_response: Dict[str, Any],
        context: Optional[Dict]
    ) -> str:
        """Generate comprehensive answer from retrieved knowledge."""
        # Start with RAG response
        base_answer = knowledge_response.get("rag_response", {}).get("answer", "")
        
        # Enhance with structured information
        structured_info = knowledge_response.get("structured_info", {})
        
        # Build comprehensive answer based on query type
        if query_type == QueryType.CONDITION_INFO:
            answer = self._build_condition_answer(base_answer, structured_info)
        elif query_type == QueryType.SYMPTOM_INFO:
            answer = self._build_symptom_answer(base_answer, structured_info)
        elif query_type == QueryType.MEDICATION_INFO:
            answer = self._build_medication_answer(base_answer, structured_info)
        else:
            answer = base_answer
        
        # Add context-specific information
        if context:
            answer = self._add_contextual_information(answer, context, query_type)
        
        return answer
    
    def _build_condition_answer(self, base_answer: str, structured_info: Dict) -> str:
        """Build comprehensive condition information answer."""
        sections = []
        
        if base_answer:
            sections.append(f"**Overview:**\n{base_answer}")
        
        for condition, info in structured_info.items():
            if info:
                sections.append(f"\n**{condition}:**")
                
                if "symptoms" in info:
                    sections.append(f"**Symptoms:** {', '.join(info['symptoms'])}")
                
                if "causes" in info:
                    sections.append(f"**Causes:** {', '.join(info['causes'])}")
                
                if "treatment" in info:
                    sections.append(f"**Treatment:** {info['treatment']}")
        
        return "\n".join(sections)
    
    def _build_symptom_answer(self, base_answer: str, structured_info: Dict) -> str:
        """Build comprehensive symptom information answer."""
        sections = []
        
        if base_answer:
            sections.append(f"**About this symptom:**\n{base_answer}")
        
        for symptom, info in structured_info.items():
            if info:
                sections.append(f"\n**{symptom}:**")
                
                if "possible_causes" in info:
                    sections.append(f"**Possible causes:** {', '.join(info['possible_causes'])}")
                
                if "when_to_worry" in info:
                    sections.append(f"**When to seek medical attention:** {', '.join(info['when_to_worry'])}")
        
        return "\n".join(sections)
    
    def _build_medication_answer(self, base_answer: str, structured_info: Dict) -> str:
        """Build comprehensive medication information answer."""
        sections = []
        
        if base_answer:
            sections.append(f"**Medication Information:**\n{base_answer}")
        
        for medication, info in structured_info.items():
            if info:
                sections.append(f"\n**{medication}:**")
                
                if "uses" in info:
                    sections.append(f"**Uses:** {', '.join(info['uses'])}")
                
                if "side_effects" in info:
                    sections.append(f"**Common side effects:** {', '.join(info['side_effects'])}")
        
        return "\n".join(sections)
    
    def _add_contextual_information(
        self,
        answer: str,
        context: Dict,
        query_type: QueryType
    ) -> str:
        """Add context-specific information to answer."""
        contextual_info = []
        
        # Age-specific information
        if "age" in context:
            age = context["age"]
            if age < 18:
                contextual_info.append("**Note for pediatric patients:** Dosages and treatments may differ for children.")
            elif age > 65:
                contextual_info.append("**Note for elderly patients:** Special considerations may apply for older adults.")
        
        # Gender-specific information
        if "gender" in context and context["gender"] == "female":
            if query_type == QueryType.MEDICATION_INFO:
                contextual_info.append("**Note:** Consider pregnancy and breastfeeding status when taking medications.")
        
        if contextual_info:
            answer += "\n\n" + "\n".join(contextual_info)
        
        return answer
    
    def _find_related_topics(self, entities: List[str], query_type: QueryType) -> List[str]:
        """Find related medical topics."""
        related_topics = []
        
        for entity in entities:
            # Use SNOMED CT relationships to find related concepts
            related_concepts = self.snomed_loader.get_related_concepts(entity, limit=5)
            
            for concept in related_concepts:
                if concept["preferred_term"] not in entities:
                    related_topics.append(concept["preferred_term"])
        
        return related_topics[:10]  # Limit to top 10
    
    def _map_to_snomed_codes(self, entities: List[str]) -> List[str]:
        """Map medical entities to SNOMED CT codes."""
        snomed_codes = []
        
        for entity in entities:
            concepts = self.snomed_loader.search_concepts(entity, limit=3)
            
            for concept in concepts:
                if concept["confidence"] > 0.8:
                    snomed_codes.append(concept["concept_id"])
        
        return snomed_codes
    
    def _get_additional_resources(
        self,
        query_type: QueryType,
        entities: List[str]
    ) -> List[Dict[str, str]]:
        """Get additional educational resources."""
        resources = []
        
        # NHS resources
        resources.append({
            "title": "NHS Health Information",
            "url": "https://www.nhs.uk/conditions/",
            "description": "Official NHS health information and guidance"
        })
        
        # Patient.info resources
        resources.append({
            "title": "Patient Information",
            "url": "https://patient.info/",
            "description": "Trusted medical information for patients"
        })
        
        # Condition-specific resources
        if query_type == QueryType.CONDITION_INFO:
            resources.append({
                "title": "Medical Charities",
                "url": "https://www.associationofmedicalresearchcharities.org.uk/",
                "description": "Find condition-specific support organizations"
            })
        
        return resources
    
    def _parse_condition_information(self, rag_response: Dict, condition: str) -> Dict[str, Any]:
        """Parse condition information from RAG response."""
        # This would parse the RAG response to extract structured information
        # For now, return basic structure
        return {
            "name": condition,
            "definition": rag_response.get("answer", ""),
            "symptoms": [],
            "causes": [],
            "treatment_options": [],
            "prognosis": "",
            "prevention": []
        }
    
    def _parse_symptom_information(self, rag_response: Dict, symptom: str) -> Dict[str, Any]:
        """Parse symptom information from RAG response."""
        return {
            "symptom": symptom,
            "description": rag_response.get("answer", ""),
            "possible_causes": [],
            "when_to_worry": [],
            "home_remedies": []
        }
    
    def _parse_medication_information(self, rag_response: Dict, medication: str) -> Dict[str, Any]:
        """Parse medication information from RAG response."""
        return {
            "name": medication,
            "uses": [],
            "side_effects": {"common": [], "serious": []},
            "dosage": "",
            "interactions": []
        }
    
    def _parse_test_information(self, rag_response: Dict, test: str) -> Dict[str, Any]:
        """Parse diagnostic test information from RAG response."""
        return {
            "test_name": test,
            "purpose": rag_response.get("answer", ""),
            "preparation": [],
            "procedure": "",
            "results": {"normal_values": "", "interpretation": ""}
        }
    
    def _parse_prevention_information(self, rag_response: Dict, condition: str) -> Dict[str, Any]:
        """Parse prevention information from RAG response."""
        return {
            "condition": condition,
            "primary_prevention": [],
            "lifestyle_modifications": [],
            "screening_recommendations": [],
            "risk_reduction": []
        }


# Example usage
def get_diabetes_information() -> MedicalKnowledgeResponse:
    """Example function to get diabetes information."""
    knowledge_system = MedicalKnowledgeSystem(
        knowledge_base_path="data/processed/medical_knowledge.json",
        snomed_path="data/ontologies/snomed",
        rag_system_path="data/rag/medical_index"
    )
    
    response = knowledge_system.answer_medical_query(
        "What is diabetes and how is it treated?",
        context={"age": 45, "gender": "male"}
    )
    
    return response
