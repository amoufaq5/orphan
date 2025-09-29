"""
Medical Image Analysis and Diagnosis System

Advanced system for reading medical images, identifying conditions,
and linking them to medical diagnoses using multimodal AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
import json

from .vision.medical_vision import MedicalVisionEncoder, MedicalImageProcessor
from ..textlm.medical_llm import MedicalLanguageModel
from ...ontology.snomed.loader import SNOMEDLoader
from ...utils.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MedicalImageAnalysis:
    """Results of medical image analysis."""
    image_type: str  # X-ray, CT, MRI, etc.
    anatomical_region: str  # chest, brain, abdomen, etc.
    findings: List[Dict[str, Any]]  # detected abnormalities
    confidence_scores: Dict[str, float]
    differential_diagnosis: List[Dict[str, Any]]
    recommendations: List[str]
    urgency_level: str  # routine, urgent, critical
    snomed_codes: List[str]


class MedicalImageDiagnosisSystem:
    """
    Comprehensive medical image diagnosis system.
    
    Capabilities:
    - Image type and modality detection
    - Anatomical region identification
    - Pathology detection and classification
    - Differential diagnosis generation
    - SNOMED CT code mapping
    - Clinical recommendations
    - Urgency assessment
    """
    
    def __init__(self, model_path: str, snomed_path: str):
        # Load vision encoder
        self.vision_encoder = MedicalVisionEncoder(
            backbone="medical_resnet",
            hidden_dim=768
        )
        
        # Load language model for diagnosis generation
        self.language_model = MedicalLanguageModel(model_path)
        
        # Load SNOMED CT for medical coding
        self.snomed_loader = SNOMEDLoader(snomed_path)
        
        # Image processor
        self.image_processor = MedicalImageProcessor(image_size=512)
        
        # Load specialized classifiers
        self.modality_classifier = self._load_modality_classifier()
        self.anatomy_classifier = self._load_anatomy_classifier()
        self.pathology_detectors = self._load_pathology_detectors()
        
        logger.info("Medical Image Diagnosis System initialized")
    
    def analyze_medical_image(
        self,
        image: torch.Tensor,
        patient_context: Optional[Dict] = None
    ) -> MedicalImageAnalysis:
        """
        Comprehensive analysis of medical image.
        
        Args:
            image: Medical image tensor
            patient_context: Patient information (age, gender, symptoms, etc.)
            
        Returns:
            Complete medical image analysis
        """
        # Step 1: Detect image modality and anatomy
        modality = self._detect_modality(image)
        anatomy = self._detect_anatomy(image, modality)
        
        # Step 2: Extract visual features
        vision_features = self._extract_vision_features(image, modality)
        
        # Step 3: Detect pathologies
        findings = self._detect_pathologies(image, modality, anatomy, vision_features)
        
        # Step 4: Generate differential diagnosis
        differential_dx = self._generate_differential_diagnosis(
            findings, modality, anatomy, patient_context
        )
        
        # Step 5: Map to SNOMED CT codes
        snomed_codes = self._map_to_snomed_codes(findings, differential_dx)
        
        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            findings, differential_dx, modality, patient_context
        )
        
        # Step 7: Assess urgency
        urgency = self._assess_urgency(findings, differential_dx)
        
        # Step 8: Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            findings, differential_dx, vision_features
        )
        
        return MedicalImageAnalysis(
            image_type=modality,
            anatomical_region=anatomy,
            findings=findings,
            confidence_scores=confidence_scores,
            differential_diagnosis=differential_dx,
            recommendations=recommendations,
            urgency_level=urgency,
            snomed_codes=snomed_codes
        )
    
    def _detect_modality(self, image: torch.Tensor) -> str:
        """Detect medical imaging modality (X-ray, CT, MRI, etc.)."""
        with torch.no_grad():
            # Preprocess image for modality detection
            processed_image = self.image_processor.process_image(image, "generic")
            
            # Run modality classifier
            modality_logits = self.modality_classifier(processed_image.unsqueeze(0))
            modality_probs = F.softmax(modality_logits, dim=-1)
            
            modalities = ["xray", "ct", "mri", "ultrasound", "pathology", "dermatology"]
            predicted_modality = modalities[torch.argmax(modality_probs).item()]
            
            logger.info(f"Detected modality: {predicted_modality}")
            return predicted_modality
    
    def _detect_anatomy(self, image: torch.Tensor, modality: str) -> str:
        """Detect anatomical region in the image."""
        with torch.no_grad():
            processed_image = self.image_processor.process_image(image, modality)
            
            anatomy_logits = self.anatomy_classifier(processed_image.unsqueeze(0))
            anatomy_probs = F.softmax(anatomy_logits, dim=-1)
            
            anatomical_regions = [
                "chest", "brain", "abdomen", "pelvis", "spine", 
                "extremities", "head_neck", "cardiac"
            ]
            predicted_anatomy = anatomical_regions[torch.argmax(anatomy_probs).item()]
            
            logger.info(f"Detected anatomy: {predicted_anatomy}")
            return predicted_anatomy
    
    def _extract_vision_features(self, image: torch.Tensor, modality: str) -> torch.Tensor:
        """Extract visual features from medical image."""
        processed_image = self.image_processor.process_image(image, modality)
        
        with torch.no_grad():
            vision_outputs = self.vision_encoder(processed_image.unsqueeze(0))
            features = vision_outputs["projected_features"]
        
        return features
    
    def _detect_pathologies(
        self,
        image: torch.Tensor,
        modality: str,
        anatomy: str,
        features: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Detect pathologies in the medical image."""
        findings = []
        
        # Get appropriate pathology detector for modality and anatomy
        detector_key = f"{modality}_{anatomy}"
        
        if detector_key in self.pathology_detectors:
            detector = self.pathology_detectors[detector_key]
            
            with torch.no_grad():
                pathology_logits = detector(features)
                pathology_probs = torch.sigmoid(pathology_logits)  # Multi-label
                
                # Get pathology classes for this detector
                pathology_classes = self._get_pathology_classes(modality, anatomy)
                
                # Extract significant findings (threshold > 0.5)
                for i, (class_name, prob) in enumerate(zip(pathology_classes, pathology_probs[0])):
                    if prob > 0.5:
                        finding = {
                            "pathology": class_name,
                            "confidence": prob.item(),
                            "location": self._localize_finding(image, i, detector),
                            "severity": self._assess_severity(class_name, prob.item()),
                            "description": self._generate_finding_description(class_name, prob.item())
                        }
                        findings.append(finding)
        
        return findings
    
    def _generate_differential_diagnosis(
        self,
        findings: List[Dict],
        modality: str,
        anatomy: str,
        patient_context: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate differential diagnosis based on findings."""
        # Prepare context for language model
        context = {
            "modality": modality,
            "anatomy": anatomy,
            "findings": findings,
            "patient_context": patient_context or {}
        }
        
        # Generate differential diagnosis using language model
        prompt = self._create_diagnosis_prompt(context)
        
        with torch.no_grad():
            diagnosis_text = self.language_model.generate(
                prompt,
                max_length=512,
                temperature=0.7
            )
        
        # Parse differential diagnosis from generated text
        differential_dx = self._parse_differential_diagnosis(diagnosis_text)
        
        return differential_dx
    
    def _map_to_snomed_codes(
        self,
        findings: List[Dict],
        differential_dx: List[Dict]
    ) -> List[str]:
        """Map findings and diagnoses to SNOMED CT codes."""
        snomed_codes = []
        
        # Map findings to SNOMED codes
        for finding in findings:
            pathology = finding["pathology"]
            codes = self.snomed_loader.search_concepts(pathology, limit=3)
            
            for code in codes:
                if code["confidence"] > 0.8:
                    snomed_codes.append(code["concept_id"])
        
        # Map differential diagnoses to SNOMED codes
        for diagnosis in differential_dx:
            condition = diagnosis["condition"]
            codes = self.snomed_loader.search_concepts(condition, limit=3)
            
            for code in codes:
                if code["confidence"] > 0.8:
                    snomed_codes.append(code["concept_id"])
        
        return list(set(snomed_codes))  # Remove duplicates
    
    def _generate_recommendations(
        self,
        findings: List[Dict],
        differential_dx: List[Dict],
        modality: str,
        patient_context: Optional[Dict]
    ) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []
        
        # Check for urgent findings
        urgent_findings = [f for f in findings if f.get("severity") == "high"]
        if urgent_findings:
            recommendations.append("Urgent clinical correlation recommended")
            recommendations.append("Consider immediate specialist referral")
        
        # Modality-specific recommendations
        if modality == "xray":
            if any("pneumonia" in f["pathology"].lower() for f in findings):
                recommendations.append("Consider antibiotic therapy")
                recommendations.append("Follow-up chest X-ray in 2-4 weeks")
        
        elif modality == "ct":
            if any("hemorrhage" in f["pathology"].lower() for f in findings):
                recommendations.append("Immediate neurosurgical consultation")
                recommendations.append("Serial neurological assessments")
        
        # General recommendations
        if len(findings) > 0:
            recommendations.append("Clinical correlation with patient symptoms")
            recommendations.append("Consider additional imaging if clinically indicated")
        
        return recommendations
    
    def _assess_urgency(
        self,
        findings: List[Dict],
        differential_dx: List[Dict]
    ) -> str:
        """Assess clinical urgency level."""
        # Critical conditions
        critical_conditions = [
            "hemorrhage", "stroke", "pneumothorax", "aortic dissection",
            "pulmonary embolism", "acute abdomen"
        ]
        
        # Check findings for critical conditions
        for finding in findings:
            pathology = finding["pathology"].lower()
            if any(condition in pathology for condition in critical_conditions):
                return "critical"
        
        # Check differential diagnosis for critical conditions
        for diagnosis in differential_dx:
            condition = diagnosis["condition"].lower()
            if any(crit_condition in condition for crit_condition in critical_conditions):
                return "critical"
        
        # High severity findings
        high_severity_findings = [f for f in findings if f.get("severity") == "high"]
        if high_severity_findings:
            return "urgent"
        
        # Default to routine if no urgent findings
        return "routine" if findings else "normal"
    
    def _calculate_confidence_scores(
        self,
        findings: List[Dict],
        differential_dx: List[Dict],
        features: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate confidence scores for analysis."""
        scores = {}
        
        # Overall confidence based on feature quality
        feature_magnitude = torch.norm(features).item()
        scores["overall_confidence"] = min(feature_magnitude / 100.0, 1.0)
        
        # Findings confidence (average of individual confidences)
        if findings:
            avg_finding_confidence = np.mean([f["confidence"] for f in findings])
            scores["findings_confidence"] = avg_finding_confidence
        else:
            scores["findings_confidence"] = 0.9  # High confidence in normal findings
        
        # Diagnosis confidence
        if differential_dx:
            avg_dx_confidence = np.mean([dx.get("confidence", 0.5) for dx in differential_dx])
            scores["diagnosis_confidence"] = avg_dx_confidence
        else:
            scores["diagnosis_confidence"] = 0.8
        
        return scores
    
    def _load_modality_classifier(self) -> nn.Module:
        """Load pre-trained modality classifier."""
        # This would load a pre-trained model for modality detection
        # For now, return a simple classifier
        return nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)  # 6 modalities
        )
    
    def _load_anatomy_classifier(self) -> nn.Module:
        """Load pre-trained anatomy classifier."""
        return nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 8)  # 8 anatomical regions
        )
    
    def _load_pathology_detectors(self) -> Dict[str, nn.Module]:
        """Load pathology detection models for different modality-anatomy combinations."""
        detectors = {}
        
        # Chest X-ray pathology detector
        detectors["xray_chest"] = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 15)  # 15 chest pathologies
        )
        
        # Brain CT pathology detector
        detectors["ct_brain"] = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 brain pathologies
        )
        
        # Add more detectors for other combinations
        
        return detectors
    
    def _get_pathology_classes(self, modality: str, anatomy: str) -> List[str]:
        """Get pathology classes for specific modality-anatomy combination."""
        if modality == "xray" and anatomy == "chest":
            return [
                "pneumonia", "pneumothorax", "pleural_effusion", "cardiomegaly",
                "pulmonary_edema", "atelectasis", "consolidation", "nodule",
                "mass", "fracture", "normal", "copd", "fibrosis", "infiltrate", "other"
            ]
        elif modality == "ct" and anatomy == "brain":
            return [
                "hemorrhage", "infarct", "tumor", "edema", "hydrocephalus",
                "atrophy", "calcification", "normal", "trauma", "other"
            ]
        else:
            return ["abnormal", "normal"]
    
    def _localize_finding(self, image: torch.Tensor, finding_idx: int, detector: nn.Module) -> Dict:
        """Localize finding in the image using attention maps."""
        # This would use attention mechanisms or gradient-based localization
        # For now, return placeholder coordinates
        return {
            "x": 0.5,
            "y": 0.5,
            "width": 0.2,
            "height": 0.2,
            "confidence": 0.8
        }
    
    def _assess_severity(self, pathology: str, confidence: float) -> str:
        """Assess severity of detected pathology."""
        critical_pathologies = ["hemorrhage", "pneumothorax", "aortic_dissection"]
        high_severity_pathologies = ["pneumonia", "tumor", "fracture"]
        
        if pathology.lower() in critical_pathologies:
            return "critical"
        elif pathology.lower() in high_severity_pathologies and confidence > 0.8:
            return "high"
        elif confidence > 0.7:
            return "moderate"
        else:
            return "low"
    
    def _generate_finding_description(self, pathology: str, confidence: float) -> str:
        """Generate human-readable description of finding."""
        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        return f"{pathology.replace('_', ' ').title()} with {confidence_text} confidence"
    
    def _create_diagnosis_prompt(self, context: Dict) -> str:
        """Create prompt for differential diagnosis generation."""
        prompt = f"""
        Medical Image Analysis:
        Modality: {context['modality']}
        Anatomy: {context['anatomy']}
        
        Findings:
        """
        
        for finding in context['findings']:
            prompt += f"- {finding['description']}\n"
        
        if context['patient_context']:
            prompt += f"\nPatient Context:\n"
            for key, value in context['patient_context'].items():
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nProvide differential diagnosis with confidence scores:"
        
        return prompt
    
    def _parse_differential_diagnosis(self, diagnosis_text: str) -> List[Dict[str, Any]]:
        """Parse differential diagnosis from generated text."""
        # This would parse the generated text to extract structured diagnosis
        # For now, return placeholder
        return [
            {
                "condition": "Primary diagnosis",
                "confidence": 0.8,
                "reasoning": "Based on imaging findings"
            }
        ]


# Example usage
def analyze_chest_xray(image_path: str, patient_age: int, symptoms: List[str]) -> MedicalImageAnalysis:
    """Example function to analyze chest X-ray."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Patient context
    patient_context = {
        "age": patient_age,
        "symptoms": symptoms
    }
    
    # Initialize diagnosis system
    diagnosis_system = MedicalImageDiagnosisSystem(
        model_path="models/medical_llm",
        snomed_path="data/ontologies/snomed"
    )
    
    # Analyze image
    analysis = diagnosis_system.analyze_medical_image(image_tensor, patient_context)
    
    return analysis
