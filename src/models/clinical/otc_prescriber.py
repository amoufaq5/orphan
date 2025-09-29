"""
Over-the-Counter (OTC) Medication Prescriber

Safe OTC medication recommendations with contraindication checking,
dosage guidance, and safety monitoring for UK healthcare standards.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from ...ontology.snomed.loader import SNOMEDLoader
from ...protocols.safety.red_flags import RedFlagDetector
from ...utils.logging.logger import get_logger

logger = get_logger(__name__)


class OTCCategory(Enum):
    """OTC medication categories."""
    ANALGESIC = "analgesic"
    ANTIHISTAMINE = "antihistamine"
    ANTACID = "antacid"
    COUGH_COLD = "cough_cold"
    TOPICAL = "topical"
    DIGESTIVE = "digestive"
    VITAMINS = "vitamins"
    FIRST_AID = "first_aid"


@dataclass
class OTCMedication:
    """OTC medication information."""
    name: str
    generic_name: str
    category: OTCCategory
    indications: List[str]
    contraindications: List[str]
    age_restrictions: Dict[str, str]  # age_group -> restriction
    dosage_adult: str
    dosage_child: str
    max_daily_dose: str
    duration_limit: str
    warnings: List[str]
    interactions: List[str]
    pregnancy_category: str
    breastfeeding_safe: bool
    active_ingredients: List[str]
    snomed_codes: List[str]


@dataclass
class OTCRecommendation:
    """OTC medication recommendation."""
    medication: OTCMedication
    recommended_dosage: str
    duration: str
    instructions: List[str]
    warnings: List[str]
    monitoring_advice: List[str]
    when_to_seek_help: List[str]
    confidence: float


class OTCPrescriber:
    """
    Safe OTC medication prescriber with comprehensive safety checks.
    
    Features:
    - Symptom-based OTC recommendations
    - Age-appropriate dosing
    - Contraindication checking
    - Drug interaction screening
    - Pregnancy/breastfeeding safety
    - UK regulatory compliance
    - Safety monitoring guidance
    """
    
    def __init__(self, otc_database_path: str, snomed_path: str):
        self.otc_database = self._load_otc_database(otc_database_path)
        self.snomed_loader = SNOMEDLoader(snomed_path)
        self.red_flag_detector = RedFlagDetector()
        
        # UK-specific OTC regulations
        self.uk_otc_regulations = self._load_uk_regulations()
        
        logger.info("OTC Prescriber initialized with UK regulations")
    
    def recommend_otc_medication(
        self,
        symptoms: List[str],
        patient_age: int,
        patient_gender: str,
        current_medications: List[str] = None,
        medical_conditions: List[str] = None,
        pregnancy_status: bool = False,
        breastfeeding_status: bool = False
    ) -> List[OTCRecommendation]:
        """
        Recommend safe OTC medications based on symptoms and patient profile.
        
        Args:
            symptoms: List of patient symptoms
            patient_age: Patient age in years
            patient_gender: Patient gender
            current_medications: Current medications
            medical_conditions: Known medical conditions
            pregnancy_status: Is patient pregnant
            breastfeeding_status: Is patient breastfeeding
            
        Returns:
            List of safe OTC recommendations
        """
        # Safety check - detect red flags
        symptom_text = " ".join(symptoms)
        red_flags = self.red_flag_detector.detect_red_flags(symptom_text)
        
        if red_flags:
            logger.warning(f"Red flags detected: {[rf.description for rf in red_flags]}")
            return []  # No OTC recommendations for red flag symptoms
        
        # Map symptoms to potential OTC medications
        candidate_medications = self._map_symptoms_to_otc(symptoms)
        
        # Filter by safety criteria
        safe_medications = []
        
        for medication in candidate_medications:
            safety_check = self._perform_safety_check(
                medication=medication,
                patient_age=patient_age,
                patient_gender=patient_gender,
                current_medications=current_medications or [],
                medical_conditions=medical_conditions or [],
                pregnancy_status=pregnancy_status,
                breastfeeding_status=breastfeeding_status
            )
            
            if safety_check["safe"]:
                recommendation = self._create_recommendation(
                    medication=medication,
                    patient_age=patient_age,
                    symptoms=symptoms,
                    safety_info=safety_check
                )
                safe_medications.append(recommendation)
        
        # Sort by confidence and safety
        safe_medications.sort(key=lambda x: x.confidence, reverse=True)
        
        return safe_medications[:3]  # Return top 3 recommendations
    
    def _load_otc_database(self, database_path: str) -> Dict[str, OTCMedication]:
        """Load comprehensive OTC medication database."""
        # This would load from the processed Kaggle OTC datasets
        otc_medications = {}
        
        # Common UK OTC medications
        medications_data = [
            {
                "name": "Paracetamol",
                "generic_name": "acetaminophen",
                "category": OTCCategory.ANALGESIC,
                "indications": ["headache", "fever", "muscle pain", "toothache"],
                "contraindications": ["severe liver disease", "alcohol dependency"],
                "age_restrictions": {"under_3_months": "prescription_only"},
                "dosage_adult": "500-1000mg every 4-6 hours",
                "dosage_child": "10-15mg/kg every 4-6 hours",
                "max_daily_dose": "4g (adults), 60mg/kg (children)",
                "duration_limit": "3 days for fever, 10 days for pain",
                "warnings": ["Do not exceed maximum dose", "Check other medications for paracetamol"],
                "interactions": ["warfarin", "carbamazepine"],
                "pregnancy_category": "A",
                "breastfeeding_safe": True,
                "active_ingredients": ["paracetamol"],
                "snomed_codes": ["387517004"]
            },
            {
                "name": "Ibuprofen",
                "generic_name": "ibuprofen",
                "category": OTCCategory.ANALGESIC,
                "indications": ["headache", "muscle pain", "inflammation", "fever"],
                "contraindications": ["peptic ulcer", "severe heart failure", "severe kidney disease"],
                "age_restrictions": {"under_6_months": "prescription_only"},
                "dosage_adult": "200-400mg every 4-6 hours",
                "dosage_child": "5-10mg/kg every 6-8 hours",
                "max_daily_dose": "1.2g (adults), 30mg/kg (children)",
                "duration_limit": "3 days for fever, 10 days for pain",
                "warnings": ["Take with food", "Avoid in pregnancy (3rd trimester)"],
                "interactions": ["warfarin", "ACE inhibitors", "diuretics"],
                "pregnancy_category": "C",
                "breastfeeding_safe": True,
                "active_ingredients": ["ibuprofen"],
                "snomed_codes": ["387207008"]
            },
            {
                "name": "Cetirizine",
                "generic_name": "cetirizine",
                "category": OTCCategory.ANTIHISTAMINE,
                "indications": ["hay fever", "allergic rhinitis", "urticaria", "itching"],
                "contraindications": ["severe kidney disease"],
                "age_restrictions": {"under_2_years": "prescription_only"},
                "dosage_adult": "10mg once daily",
                "dosage_child": "2.5-5mg once daily (age dependent)",
                "max_daily_dose": "10mg (adults), 5mg (children)",
                "duration_limit": "Continuous use as needed",
                "warnings": ["May cause drowsiness", "Avoid alcohol"],
                "interactions": ["sedatives", "alcohol"],
                "pregnancy_category": "B",
                "breastfeeding_safe": True,
                "active_ingredients": ["cetirizine hydrochloride"],
                "snomed_codes": ["387467008"]
            },
            {
                "name": "Gaviscon",
                "generic_name": "alginate_antacid",
                "category": OTCCategory.ANTACID,
                "indications": ["heartburn", "acid reflux", "indigestion"],
                "contraindications": ["phenylketonuria (some formulations)"],
                "age_restrictions": {"under_12_years": "check_formulation"},
                "dosage_adult": "10-20ml after meals and bedtime",
                "dosage_child": "5-10ml (age dependent)",
                "max_daily_dose": "80ml",
                "duration_limit": "2 weeks continuous use",
                "warnings": ["Contains sodium", "May affect other medication absorption"],
                "interactions": ["tetracyclines", "iron supplements"],
                "pregnancy_category": "A",
                "breastfeeding_safe": True,
                "active_ingredients": ["sodium alginate", "sodium bicarbonate", "calcium carbonate"],
                "snomed_codes": ["396064000"]
            },
            {
                "name": "Loratadine",
                "generic_name": "loratadine",
                "category": OTCCategory.ANTIHISTAMINE,
                "indications": ["hay fever", "allergic rhinitis", "chronic urticaria"],
                "contraindications": ["severe liver disease"],
                "age_restrictions": {"under_2_years": "prescription_only"},
                "dosage_adult": "10mg once daily",
                "dosage_child": "5mg once daily (2-12 years)",
                "max_daily_dose": "10mg",
                "duration_limit": "Continuous use as needed",
                "warnings": ["Non-drowsy formulation", "Take on empty stomach"],
                "interactions": ["ketoconazole", "erythromycin"],
                "pregnancy_category": "B",
                "breastfeeding_safe": True,
                "active_ingredients": ["loratadine"],
                "snomed_codes": ["386884004"]
            }
        ]
        
        for med_data in medications_data:
            medication = OTCMedication(**med_data)
            otc_medications[medication.name.lower()] = medication
        
        return otc_medications
    
    def _load_uk_regulations(self) -> Dict[str, Any]:
        """Load UK-specific OTC regulations and guidelines."""
        return {
            "max_pack_sizes": {
                "paracetamol": {"pharmacy": 100, "general_sale": 16},
                "ibuprofen": {"pharmacy": 84, "general_sale": 16},
                "aspirin": {"pharmacy": 100, "general_sale": 16}
            },
            "age_restrictions": {
                "aspirin": {"min_age": 16, "reason": "Reye's syndrome risk"},
                "ibuprofen": {"min_age": 0.5, "reason": "Safety in infants"},
                "paracetamol": {"min_age": 0.25, "reason": "Dosing accuracy"}
            },
            "pregnancy_restrictions": {
                "ibuprofen": {"trimester_3": "avoid"},
                "aspirin": {"all_trimesters": "avoid_high_dose"},
                "codeine": {"all_trimesters": "avoid"}
            }
        }
    
    def _map_symptoms_to_otc(self, symptoms: List[str]) -> List[OTCMedication]:
        """Map patient symptoms to potential OTC medications."""
        candidate_medications = []
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            
            for medication in self.otc_database.values():
                # Check if symptom matches any indication
                for indication in medication.indications:
                    if (indication.lower() in symptom_lower or 
                        symptom_lower in indication.lower()):
                        candidate_medications.append(medication)
        
        # Remove duplicates
        unique_medications = []
        seen_names = set()
        
        for med in candidate_medications:
            if med.name not in seen_names:
                unique_medications.append(med)
                seen_names.add(med.name)
        
        return unique_medications
    
    def _perform_safety_check(
        self,
        medication: OTCMedication,
        patient_age: int,
        patient_gender: str,
        current_medications: List[str],
        medical_conditions: List[str],
        pregnancy_status: bool,
        breastfeeding_status: bool
    ) -> Dict[str, Any]:
        """Perform comprehensive safety check for OTC medication."""
        safety_issues = []
        warnings = []
        
        # Age restrictions
        if patient_age < 18:
            age_group = self._get_age_group(patient_age)
            if age_group in medication.age_restrictions:
                restriction = medication.age_restrictions[age_group]
                if restriction == "prescription_only":
                    safety_issues.append(f"Prescription required for age {patient_age}")
        
        # Pregnancy safety
        if pregnancy_status:
            if medication.pregnancy_category in ["D", "X"]:
                safety_issues.append("Not safe in pregnancy")
            elif medication.pregnancy_category == "C":
                warnings.append("Use with caution in pregnancy - consult healthcare provider")
        
        # Breastfeeding safety
        if breastfeeding_status and not medication.breastfeeding_safe:
            safety_issues.append("Not recommended while breastfeeding")
        
        # Contraindications
        for condition in medical_conditions:
            for contraindication in medication.contraindications:
                if condition.lower() in contraindication.lower():
                    safety_issues.append(f"Contraindicated with {condition}")
        
        # Drug interactions
        for current_med in current_medications:
            for interaction in medication.interactions:
                if current_med.lower() in interaction.lower():
                    warnings.append(f"Potential interaction with {current_med}")
        
        # UK-specific regulations
        uk_restrictions = self._check_uk_restrictions(medication, patient_age)
        if uk_restrictions:
            warnings.extend(uk_restrictions)
        
        return {
            "safe": len(safety_issues) == 0,
            "safety_issues": safety_issues,
            "warnings": warnings,
            "confidence": 1.0 - (len(safety_issues) * 0.5 + len(warnings) * 0.1)
        }
    
    def _get_age_group(self, age_years: int) -> str:
        """Get age group classification."""
        if age_years < 0.25:
            return "under_3_months"
        elif age_years < 0.5:
            return "under_6_months"
        elif age_years < 2:
            return "under_2_years"
        elif age_years < 12:
            return "under_12_years"
        elif age_years < 16:
            return "under_16_years"
        elif age_years < 18:
            return "under_18_years"
        else:
            return "adult"
    
    def _check_uk_restrictions(self, medication: OTCMedication, patient_age: int) -> List[str]:
        """Check UK-specific OTC restrictions."""
        warnings = []
        
        med_name = medication.name.lower()
        
        # Age-based restrictions
        if med_name in self.uk_otc_regulations["age_restrictions"]:
            restriction = self.uk_otc_regulations["age_restrictions"][med_name]
            if patient_age < restriction["min_age"]:
                warnings.append(f"UK regulation: {restriction['reason']}")
        
        return warnings
    
    def _create_recommendation(
        self,
        medication: OTCMedication,
        patient_age: int,
        symptoms: List[str],
        safety_info: Dict[str, Any]
    ) -> OTCRecommendation:
        """Create detailed OTC recommendation."""
        # Determine appropriate dosage
        if patient_age >= 18:
            recommended_dosage = medication.dosage_adult
        else:
            recommended_dosage = medication.dosage_child
        
        # Generate instructions
        instructions = [
            f"Take {recommended_dosage}",
            f"Maximum daily dose: {medication.max_daily_dose}",
            f"Duration limit: {medication.duration_limit}"
        ]
        
        # Add medication-specific instructions
        if "food" in " ".join(medication.warnings).lower():
            instructions.append("Take with food to reduce stomach irritation")
        
        # Monitoring advice
        monitoring_advice = [
            "Monitor for symptom improvement",
            "Stop if symptoms worsen or new symptoms develop",
            "Do not exceed recommended dose or duration"
        ]
        
        # When to seek help
        when_to_seek_help = [
            "Symptoms persist beyond recommended duration",
            "Symptoms worsen despite treatment",
            "New or concerning symptoms develop",
            "Signs of allergic reaction (rash, swelling, difficulty breathing)"
        ]
        
        # Add symptom-specific advice
        if any("pain" in symptom.lower() for symptom in symptoms):
            when_to_seek_help.append("Pain becomes severe or unbearable")
        
        if any("fever" in symptom.lower() for symptom in symptoms):
            when_to_seek_help.append("Fever exceeds 39°C (102°F) or persists >3 days")
        
        return OTCRecommendation(
            medication=medication,
            recommended_dosage=recommended_dosage,
            duration=medication.duration_limit,
            instructions=instructions,
            warnings=medication.warnings + safety_info["warnings"],
            monitoring_advice=monitoring_advice,
            when_to_seek_help=when_to_seek_help,
            confidence=safety_info["confidence"]
        )
    
    def check_otc_interactions(
        self,
        otc_medication: str,
        current_medications: List[str]
    ) -> Dict[str, Any]:
        """Check for interactions between OTC and prescription medications."""
        interactions = []
        
        if otc_medication.lower() in self.otc_database:
            medication = self.otc_database[otc_medication.lower()]
            
            for current_med in current_medications:
                for interaction in medication.interactions:
                    if current_med.lower() in interaction.lower():
                        interactions.append({
                            "medication": current_med,
                            "interaction": interaction,
                            "severity": self._assess_interaction_severity(interaction),
                            "advice": self._get_interaction_advice(interaction)
                        })
        
        return {
            "has_interactions": len(interactions) > 0,
            "interactions": interactions,
            "recommendation": "Consult pharmacist or GP" if interactions else "No known interactions"
        }
    
    def _assess_interaction_severity(self, interaction: str) -> str:
        """Assess severity of drug interaction."""
        high_risk_interactions = ["warfarin", "lithium", "methotrexate"]
        moderate_risk_interactions = ["ace inhibitors", "diuretics", "antidepressants"]
        
        interaction_lower = interaction.lower()
        
        if any(drug in interaction_lower for drug in high_risk_interactions):
            return "high"
        elif any(drug in interaction_lower for drug in moderate_risk_interactions):
            return "moderate"
        else:
            return "low"
    
    def _get_interaction_advice(self, interaction: str) -> str:
        """Get advice for managing drug interaction."""
        advice_map = {
            "warfarin": "Monitor INR more frequently, consult anticoagulation clinic",
            "ace inhibitors": "Monitor blood pressure and kidney function",
            "diuretics": "Monitor blood pressure and electrolytes",
            "antidepressants": "Monitor for increased side effects"
        }
        
        for drug, advice in advice_map.items():
            if drug in interaction.lower():
                return advice
        
        return "Consult healthcare provider for guidance"


# Example usage
def recommend_otc_for_headache(patient_age: int, pregnancy_status: bool = False) -> List[OTCRecommendation]:
    """Example function to recommend OTC for headache."""
    prescriber = OTCPrescriber(
        otc_database_path="data/processed/otc_database.json",
        snomed_path="data/ontologies/snomed"
    )
    
    recommendations = prescriber.recommend_otc_medication(
        symptoms=["headache"],
        patient_age=patient_age,
        patient_gender="female",
        pregnancy_status=pregnancy_status
    )
    
    return recommendations
