"""
Patient Profiling and Long-term Case Management System

Comprehensive system for creating patient profiles, tracking medical history,
monitoring health trends, and managing long-term care continuity.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from ...protocols.safety.red_flags import RedFlagDetector
from ...ontology.snomed.loader import SNOMEDLoader
from ...utils.logging.logger import get_logger

logger = get_logger(__name__)


class InteractionType(Enum):
    """Types of patient interactions."""
    CONSULTATION = "consultation"
    SYMPTOM_CHECK = "symptom_check"
    MEDICATION_QUERY = "medication_query"
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    HEALTH_EDUCATION = "health_education"
    PREVENTIVE_CARE = "preventive_care"


class RiskLevel(Enum):
    """Patient risk levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatientDemographics:
    """Patient demographic information."""
    patient_id: str
    age: int
    gender: str
    date_of_birth: str
    postcode: str  # UK postcode for regional health data
    ethnicity: Optional[str] = None
    occupation: Optional[str] = None
    emergency_contact: Optional[Dict[str, str]] = None


@dataclass
class MedicalHistory:
    """Patient medical history."""
    chronic_conditions: List[str]
    past_surgeries: List[Dict[str, str]]
    allergies: List[Dict[str, str]]
    family_history: List[Dict[str, str]]
    immunizations: List[Dict[str, str]]
    previous_hospitalizations: List[Dict[str, str]]
    mental_health_history: List[str]


@dataclass
class CurrentMedications:
    """Current patient medications."""
    prescription_medications: List[Dict[str, Any]]
    otc_medications: List[Dict[str, Any]]
    supplements: List[Dict[str, Any]]
    medication_adherence: Dict[str, float]  # medication -> adherence score


@dataclass
class LifestyleFactors:
    """Patient lifestyle information."""
    smoking_status: str
    alcohol_consumption: str
    exercise_frequency: str
    diet_type: str
    sleep_patterns: Dict[str, Any]
    stress_level: int  # 1-10 scale
    social_support: str


@dataclass
class VitalSigns:
    """Patient vital signs and measurements."""
    timestamp: str
    blood_pressure: Optional[Dict[str, int]] = None  # systolic, diastolic
    heart_rate: Optional[int] = None
    temperature: Optional[float] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    bmi: Optional[float] = None
    oxygen_saturation: Optional[int] = None


@dataclass
class Interaction:
    """Individual patient interaction record."""
    interaction_id: str
    timestamp: str
    interaction_type: InteractionType
    chief_complaint: str
    symptoms: List[str]
    assessment: Dict[str, Any]
    recommendations: List[str]
    red_flags_detected: List[str]
    follow_up_required: bool
    follow_up_date: Optional[str]
    satisfaction_score: Optional[int]
    outcome: Optional[str]


@dataclass
class HealthTrends:
    """Patient health trends and analytics."""
    symptom_patterns: Dict[str, List[str]]  # symptom -> dates
    medication_effectiveness: Dict[str, float]
    health_score_trend: List[Tuple[str, float]]  # date, score
    risk_factors: List[str]
    improvement_areas: List[str]
    care_gaps: List[str]


@dataclass
class PatientProfile:
    """Comprehensive patient profile."""
    demographics: PatientDemographics
    medical_history: MedicalHistory
    current_medications: CurrentMedications
    lifestyle_factors: LifestyleFactors
    interactions: List[Interaction]
    vital_signs: List[VitalSigns]
    health_trends: HealthTrends
    risk_level: RiskLevel
    care_plan: Dict[str, Any]
    created_date: str
    last_updated: str


class PatientProfilingSystem:
    """
    Comprehensive patient profiling and case management system.
    
    Features:
    - Complete patient profiles with medical history
    - Long-term interaction tracking
    - Health trend analysis
    - Risk stratification
    - Care plan management
    - Medication adherence monitoring
    - Preventive care reminders
    - Outcome tracking
    - Care continuity
    - Population health insights
    """
    
    def __init__(self, database_path: str, snomed_path: str):
        self.database_path = database_path
        self.patient_profiles = self._load_patient_database()
        self.snomed_loader = SNOMEDLoader(snomed_path)
        self.red_flag_detector = RedFlagDetector()
        
        logger.info("Patient Profiling System initialized")
    
    def create_patient_profile(
        self,
        demographics: PatientDemographics,
        medical_history: Optional[MedicalHistory] = None,
        lifestyle_factors: Optional[LifestyleFactors] = None
    ) -> str:
        """Create new patient profile."""
        patient_id = demographics.patient_id or str(uuid.uuid4())
        
        # Initialize profile components
        if medical_history is None:
            medical_history = MedicalHistory(
                chronic_conditions=[],
                past_surgeries=[],
                allergies=[],
                family_history=[],
                immunizations=[],
                previous_hospitalizations=[],
                mental_health_history=[]
            )
        
        if lifestyle_factors is None:
            lifestyle_factors = LifestyleFactors(
                smoking_status="unknown",
                alcohol_consumption="unknown",
                exercise_frequency="unknown",
                diet_type="unknown",
                sleep_patterns={},
                stress_level=5,
                social_support="unknown"
            )
        
        # Create initial health trends
        health_trends = HealthTrends(
            symptom_patterns={},
            medication_effectiveness={},
            health_score_trend=[],
            risk_factors=[],
            improvement_areas=[],
            care_gaps=[]
        )
        
        # Assess initial risk level
        risk_level = self._assess_risk_level(medical_history, lifestyle_factors)
        
        # Create care plan
        care_plan = self._create_initial_care_plan(medical_history, lifestyle_factors, risk_level)
        
        # Create profile
        profile = PatientProfile(
            demographics=demographics,
            medical_history=medical_history,
            current_medications=CurrentMedications([], [], [], {}),
            lifestyle_factors=lifestyle_factors,
            interactions=[],
            vital_signs=[],
            health_trends=health_trends,
            risk_level=risk_level,
            care_plan=care_plan,
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        # Store profile
        self.patient_profiles[patient_id] = profile
        self._save_patient_database()
        
        logger.info(f"Created patient profile: {patient_id}")
        return patient_id
    
    def add_interaction(
        self,
        patient_id: str,
        interaction_type: InteractionType,
        chief_complaint: str,
        symptoms: List[str],
        assessment: Dict[str, Any],
        recommendations: List[str]
    ) -> str:
        """Add new patient interaction."""
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient profile not found: {patient_id}")
        
        profile = self.patient_profiles[patient_id]
        
        # Detect red flags
        symptom_text = f"{chief_complaint} {' '.join(symptoms)}"
        red_flags = self.red_flag_detector.detect_red_flags(symptom_text)
        red_flag_descriptions = [rf.description for rf in red_flags]
        
        # Determine follow-up requirements
        follow_up_required = len(red_flags) > 0 or interaction_type == InteractionType.EMERGENCY
        follow_up_date = None
        
        if follow_up_required:
            # Schedule follow-up based on severity
            if red_flags:
                follow_up_date = (datetime.now() + timedelta(days=1)).isoformat()
            else:
                follow_up_date = (datetime.now() + timedelta(days=7)).isoformat()
        
        # Create interaction record
        interaction = Interaction(
            interaction_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            interaction_type=interaction_type,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            assessment=assessment,
            recommendations=recommendations,
            red_flags_detected=red_flag_descriptions,
            follow_up_required=follow_up_required,
            follow_up_date=follow_up_date,
            satisfaction_score=None,
            outcome=None
        )
        
        # Add to profile
        profile.interactions.append(interaction)
        
        # Update health trends
        self._update_health_trends(profile, interaction)
        
        # Update risk assessment
        profile.risk_level = self._reassess_risk_level(profile)
        
        # Update care plan if needed
        if len(red_flags) > 0 or interaction_type == InteractionType.EMERGENCY:
            profile.care_plan = self._update_care_plan(profile, interaction)
        
        # Update timestamp
        profile.last_updated = datetime.now().isoformat()
        
        # Save changes
        self._save_patient_database()
        
        logger.info(f"Added interaction for patient {patient_id}: {interaction.interaction_id}")
        return interaction.interaction_id
    
    def add_vital_signs(
        self,
        patient_id: str,
        vital_signs: VitalSigns
    ) -> None:
        """Add vital signs measurement."""
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient profile not found: {patient_id}")
        
        profile = self.patient_profiles[patient_id]
        
        # Calculate BMI if height and weight available
        if vital_signs.height and vital_signs.weight:
            height_m = vital_signs.height / 100  # Convert cm to m
            vital_signs.bmi = round(vital_signs.weight / (height_m ** 2), 1)
        
        profile.vital_signs.append(vital_signs)
        
        # Analyze vital sign trends
        self._analyze_vital_trends(profile)
        
        profile.last_updated = datetime.now().isoformat()
        self._save_patient_database()
        
        logger.info(f"Added vital signs for patient {patient_id}")
    
    def update_medications(
        self,
        patient_id: str,
        medications: CurrentMedications
    ) -> None:
        """Update patient medications."""
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient profile not found: {patient_id}")
        
        profile = self.patient_profiles[patient_id]
        profile.current_medications = medications
        
        # Check for drug interactions
        interactions = self._check_drug_interactions(medications)
        if interactions:
            logger.warning(f"Drug interactions detected for patient {patient_id}: {interactions}")
        
        profile.last_updated = datetime.now().isoformat()
        self._save_patient_database()
        
        logger.info(f"Updated medications for patient {patient_id}")
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient summary."""
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient profile not found: {patient_id}")
        
        profile = self.patient_profiles[patient_id]
        
        # Calculate summary statistics
        total_interactions = len(profile.interactions)
        recent_interactions = len([
            i for i in profile.interactions 
            if datetime.fromisoformat(i.timestamp) > datetime.now() - timedelta(days=30)
        ])
        
        # Get recent symptoms
        recent_symptoms = []
        for interaction in profile.interactions[-5:]:  # Last 5 interactions
            recent_symptoms.extend(interaction.symptoms)
        
        # Calculate health score
        health_score = self._calculate_health_score(profile)
        
        return {
            "patient_id": patient_id,
            "demographics": asdict(profile.demographics),
            "risk_level": profile.risk_level.value,
            "health_score": health_score,
            "total_interactions": total_interactions,
            "recent_interactions_30d": recent_interactions,
            "recent_symptoms": list(set(recent_symptoms)),
            "chronic_conditions": profile.medical_history.chronic_conditions,
            "current_medications_count": len(profile.current_medications.prescription_medications),
            "last_interaction": profile.interactions[-1].timestamp if profile.interactions else None,
            "follow_up_required": any(i.follow_up_required for i in profile.interactions[-3:]),
            "care_gaps": profile.health_trends.care_gaps,
            "improvement_areas": profile.health_trends.improvement_areas
        }
    
    def get_health_trends(self, patient_id: str, days: int = 90) -> Dict[str, Any]:
        """Get patient health trends over specified period."""
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient profile not found: {patient_id}")
        
        profile = self.patient_profiles[patient_id]
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter interactions by date
        recent_interactions = [
            i for i in profile.interactions
            if datetime.fromisoformat(i.timestamp) > cutoff_date
        ]
        
        # Analyze trends
        symptom_frequency = {}
        interaction_types = {}
        
        for interaction in recent_interactions:
            # Count symptoms
            for symptom in interaction.symptoms:
                symptom_frequency[symptom] = symptom_frequency.get(symptom, 0) + 1
            
            # Count interaction types
            int_type = interaction.interaction_type.value
            interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
        
        # Vital signs trends
        recent_vitals = [
            v for v in profile.vital_signs
            if datetime.fromisoformat(v.timestamp) > cutoff_date
        ]
        
        vital_trends = self._analyze_vital_sign_trends(recent_vitals)
        
        return {
            "period_days": days,
            "total_interactions": len(recent_interactions),
            "symptom_frequency": symptom_frequency,
            "interaction_types": interaction_types,
            "vital_trends": vital_trends,
            "health_score_trend": profile.health_trends.health_score_trend[-days:],
            "medication_adherence": profile.current_medications.medication_adherence,
            "risk_factors": profile.health_trends.risk_factors
        }
    
    def get_care_recommendations(self, patient_id: str) -> Dict[str, Any]:
        """Get personalized care recommendations."""
        if patient_id not in self.patient_profiles:
            raise ValueError(f"Patient profile not found: {patient_id}")
        
        profile = self.patient_profiles[patient_id]
        
        recommendations = {
            "immediate_actions": [],
            "preventive_care": [],
            "lifestyle_modifications": [],
            "medication_reviews": [],
            "screening_due": [],
            "follow_up_appointments": []
        }
        
        # Check for immediate actions
        recent_red_flags = []
        for interaction in profile.interactions[-3:]:
            recent_red_flags.extend(interaction.red_flags_detected)
        
        if recent_red_flags:
            recommendations["immediate_actions"].append(
                "Review recent red flag symptoms with healthcare provider"
            )
        
        # Preventive care based on age and risk factors
        age = profile.demographics.age
        
        if age >= 40 and "cardiovascular_screening" not in [i.chief_complaint for i in profile.interactions[-12:]]:
            recommendations["screening_due"].append("Cardiovascular risk assessment")
        
        if age >= 50 and profile.demographics.gender == "female":
            recommendations["screening_due"].append("Mammography screening")
        
        # Lifestyle recommendations based on risk factors
        if "hypertension" in profile.medical_history.chronic_conditions:
            recommendations["lifestyle_modifications"].extend([
                "Reduce sodium intake",
                "Increase physical activity",
                "Monitor blood pressure regularly"
            ])
        
        # Medication adherence
        poor_adherence = [
            med for med, adherence in profile.current_medications.medication_adherence.items()
            if adherence < 0.8
        ]
        
        if poor_adherence:
            recommendations["medication_reviews"].append(
                f"Review adherence for: {', '.join(poor_adherence)}"
            )
        
        return recommendations
    
    def _load_patient_database(self) -> Dict[str, PatientProfile]:
        """Load patient database from storage."""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                
                # Convert dictionaries back to dataclasses
                profiles = {}
                for patient_id, profile_data in data.items():
                    # This would involve proper deserialization
                    # For now, return empty dict
                    profiles[patient_id] = profile_data
                
                return profiles
        except FileNotFoundError:
            logger.info("Patient database not found, creating new one")
            return {}
    
    def _save_patient_database(self) -> None:
        """Save patient database to storage."""
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_data = {}
        
        for patient_id, profile in self.patient_profiles.items():
            serializable_data[patient_id] = asdict(profile)
        
        with open(self.database_path, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
    
    def _assess_risk_level(
        self,
        medical_history: MedicalHistory,
        lifestyle_factors: LifestyleFactors
    ) -> RiskLevel:
        """Assess patient risk level based on history and lifestyle."""
        risk_score = 0
        
        # Chronic conditions
        high_risk_conditions = ["diabetes", "heart disease", "copd", "cancer"]
        for condition in medical_history.chronic_conditions:
            if any(hr_condition in condition.lower() for hr_condition in high_risk_conditions):
                risk_score += 3
            else:
                risk_score += 1
        
        # Lifestyle factors
        if lifestyle_factors.smoking_status == "current_smoker":
            risk_score += 2
        elif lifestyle_factors.smoking_status == "former_smoker":
            risk_score += 1
        
        if lifestyle_factors.alcohol_consumption == "heavy":
            risk_score += 2
        
        if lifestyle_factors.exercise_frequency == "sedentary":
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _reassess_risk_level(self, profile: PatientProfile) -> RiskLevel:
        """Reassess risk level based on recent interactions."""
        base_risk = self._assess_risk_level(
            profile.medical_history,
            profile.lifestyle_factors
        )
        
        # Check recent red flags
        recent_red_flags = []
        for interaction in profile.interactions[-5:]:
            recent_red_flags.extend(interaction.red_flags_detected)
        
        if recent_red_flags:
            # Escalate risk level
            if base_risk == RiskLevel.LOW:
                return RiskLevel.MODERATE
            elif base_risk == RiskLevel.MODERATE:
                return RiskLevel.HIGH
            elif base_risk == RiskLevel.HIGH:
                return RiskLevel.CRITICAL
        
        return base_risk
    
    def _create_initial_care_plan(
        self,
        medical_history: MedicalHistory,
        lifestyle_factors: LifestyleFactors,
        risk_level: RiskLevel
    ) -> Dict[str, Any]:
        """Create initial care plan."""
        care_plan = {
            "monitoring_frequency": "monthly" if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else "quarterly",
            "screening_schedule": {},
            "lifestyle_goals": [],
            "medication_management": {},
            "emergency_protocols": []
        }
        
        # Add condition-specific monitoring
        for condition in medical_history.chronic_conditions:
            if "diabetes" in condition.lower():
                care_plan["screening_schedule"]["hba1c"] = "every_3_months"
                care_plan["lifestyle_goals"].append("Blood glucose monitoring")
            
            if "hypertension" in condition.lower():
                care_plan["screening_schedule"]["blood_pressure"] = "weekly"
                care_plan["lifestyle_goals"].append("Blood pressure monitoring")
        
        return care_plan
    
    def _update_care_plan(self, profile: PatientProfile, interaction: Interaction) -> Dict[str, Any]:
        """Update care plan based on new interaction."""
        care_plan = profile.care_plan.copy()
        
        # Escalate monitoring if red flags detected
        if interaction.red_flags_detected:
            care_plan["monitoring_frequency"] = "weekly"
            care_plan["emergency_protocols"].append(
                f"Monitor for: {', '.join(interaction.red_flags_detected)}"
            )
        
        return care_plan
    
    def _update_health_trends(self, profile: PatientProfile, interaction: Interaction) -> None:
        """Update health trends based on new interaction."""
        # Update symptom patterns
        for symptom in interaction.symptoms:
            if symptom not in profile.health_trends.symptom_patterns:
                profile.health_trends.symptom_patterns[symptom] = []
            profile.health_trends.symptom_patterns[symptom].append(interaction.timestamp)
        
        # Calculate and update health score
        health_score = self._calculate_health_score(profile)
        profile.health_trends.health_score_trend.append((interaction.timestamp, health_score))
        
        # Keep only last 100 health scores
        if len(profile.health_trends.health_score_trend) > 100:
            profile.health_trends.health_score_trend = profile.health_trends.health_score_trend[-100:]
    
    def _calculate_health_score(self, profile: PatientProfile) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # Deduct for chronic conditions
        score -= len(profile.medical_history.chronic_conditions) * 5
        
        # Deduct for recent red flags
        recent_red_flags = sum(
            len(i.red_flags_detected) for i in profile.interactions[-5:]
        )
        score -= recent_red_flags * 10
        
        # Deduct for poor medication adherence
        if profile.current_medications.medication_adherence:
            avg_adherence = np.mean(list(profile.current_medications.medication_adherence.values()))
            score -= (1 - avg_adherence) * 20
        
        # Adjust for lifestyle factors
        if profile.lifestyle_factors.smoking_status == "current_smoker":
            score -= 15
        elif profile.lifestyle_factors.smoking_status == "former_smoker":
            score -= 5
        
        if profile.lifestyle_factors.exercise_frequency == "regular":
            score += 5
        elif profile.lifestyle_factors.exercise_frequency == "sedentary":
            score -= 10
        
        return max(0, min(100, score))
    
    def _analyze_vital_trends(self, profile: PatientProfile) -> None:
        """Analyze vital sign trends for health insights."""
        if len(profile.vital_signs) < 2:
            return
        
        recent_vitals = profile.vital_signs[-10:]  # Last 10 measurements
        
        # Analyze blood pressure trends
        bp_readings = [v.blood_pressure for v in recent_vitals if v.blood_pressure]
        if len(bp_readings) >= 3:
            systolic_trend = [bp["systolic"] for bp in bp_readings]
            if np.mean(systolic_trend[-3:]) > np.mean(systolic_trend[:-3]):
                profile.health_trends.risk_factors.append("Rising blood pressure trend")
        
        # Analyze weight trends
        weight_readings = [v.weight for v in recent_vitals if v.weight]
        if len(weight_readings) >= 3:
            if weight_readings[-1] > weight_readings[0] * 1.1:  # 10% weight gain
                profile.health_trends.improvement_areas.append("Weight management")
    
    def _analyze_vital_sign_trends(self, vitals: List[VitalSigns]) -> Dict[str, Any]:
        """Analyze vital sign trends over time."""
        trends = {}
        
        if not vitals:
            return trends
        
        # Blood pressure trends
        bp_readings = [v.blood_pressure for v in vitals if v.blood_pressure]
        if bp_readings:
            systolic_values = [bp["systolic"] for bp in bp_readings]
            diastolic_values = [bp["diastolic"] for bp in bp_readings]
            
            trends["blood_pressure"] = {
                "systolic_avg": np.mean(systolic_values),
                "diastolic_avg": np.mean(diastolic_values),
                "trend": "increasing" if systolic_values[-1] > systolic_values[0] else "stable"
            }
        
        # Weight trends
        weights = [v.weight for v in vitals if v.weight]
        if weights:
            trends["weight"] = {
                "current": weights[-1],
                "change": weights[-1] - weights[0] if len(weights) > 1 else 0,
                "trend": "increasing" if len(weights) > 1 and weights[-1] > weights[0] else "stable"
            }
        
        return trends
    
    def _check_drug_interactions(self, medications: CurrentMedications) -> List[str]:
        """Check for potential drug interactions."""
        interactions = []
        
        all_meds = (
            medications.prescription_medications + 
            medications.otc_medications + 
            medications.supplements
        )
        
        # Simple interaction checking (would be more sophisticated in practice)
        med_names = [med.get("name", "").lower() for med in all_meds]
        
        # Known interaction pairs
        interaction_pairs = [
            ("warfarin", "aspirin"),
            ("metformin", "alcohol"),
            ("simvastatin", "grapefruit")
        ]
        
        for med1, med2 in interaction_pairs:
            if any(med1 in name for name in med_names) and any(med2 in name for name in med_names):
                interactions.append(f"Potential interaction: {med1} and {med2}")
        
        return interactions


# Example usage
def create_sample_patient() -> str:
    """Create a sample patient profile."""
    profiling_system = PatientProfilingSystem(
        database_path="data/patients/patient_profiles.json",
        snomed_path="data/ontologies/snomed"
    )
    
    # Create demographics
    demographics = PatientDemographics(
        patient_id="",  # Will be auto-generated
        age=45,
        gender="male",
        date_of_birth="1978-05-15",
        postcode="SW1A 1AA",
        ethnicity="White British",
        occupation="Teacher"
    )
    
    # Create medical history
    medical_history = MedicalHistory(
        chronic_conditions=["Type 2 Diabetes", "Hypertension"],
        past_surgeries=[],
        allergies=[{"allergen": "Penicillin", "reaction": "Rash"}],
        family_history=[{"condition": "Heart Disease", "relation": "Father"}],
        immunizations=[],
        previous_hospitalizations=[],
        mental_health_history=[]
    )
    
    # Create lifestyle factors
    lifestyle_factors = LifestyleFactors(
        smoking_status="former_smoker",
        alcohol_consumption="moderate",
        exercise_frequency="occasional",
        diet_type="standard",
        sleep_patterns={"hours": 7, "quality": "fair"},
        stress_level=6,
        social_support="good"
    )
    
    # Create profile
    patient_id = profiling_system.create_patient_profile(
        demographics, medical_history, lifestyle_factors
    )
    
    return patient_id
