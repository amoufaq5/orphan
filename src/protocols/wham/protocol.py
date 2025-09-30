"""
WHAM Protocol Implementation for Medical Consultations

WHAM Protocol: Who, How, Action, Monitoring
- Who: Patient demographics and context
- How: Symptom presentation and characteristics  
- Action: Treatment recommendations
- Monitoring: Follow-up and safety checks
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ...utils.logging.logger import get_logger
from ...ontology.snomed.loader import SNOMEDLoader
from ..safety.red_flags import RedFlagDetector

logger = get_logger(__name__)


class WHAMStage(Enum):
    """WWHAM protocol stages (Who, What, How, Action, Medication, Monitoring)."""
    WHO = "who"
    WHAT = "what"
    HOW = "how" 
    ACTION = "action"
    MEDICATION = "medication"
    MONITORING = "monitoring"


@dataclass
class PatientContext:
    """Patient context for WHAM protocol."""
    age: Optional[int] = None
    gender: Optional[str] = None
    pregnancy_status: Optional[bool] = None
    breastfeeding: Optional[bool] = None
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    occupation: Optional[str] = None
    lifestyle_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SymptomPresentation:
    """Symptom presentation details."""
    primary_complaint: str = ""
    duration: Optional[str] = None
    severity: Optional[int] = None  # 1-10 scale
    location: Optional[str] = None
    quality: Optional[str] = None
    timing: Optional[str] = None
    aggravating_factors: List[str] = field(default_factory=list)
    relieving_factors: List[str] = field(default_factory=list)
    associated_symptoms: List[str] = field(default_factory=list)
    previous_episodes: Optional[bool] = None
    impact_on_daily_life: Optional[str] = None


@dataclass
class WHAMRecommendation:
    """WHAM protocol recommendation."""
    category: str  # "self_care", "otc_medication", "prescription", "referral"
    recommendation: str
    rationale: str
    safety_considerations: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    monitoring_required: bool = False


@dataclass
class WHAMAssessment:
    """Complete WHAM assessment."""
    patient_context: PatientContext
    symptom_presentation: SymptomPresentation
    recommendations: List[WHAMRecommendation] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    referral_urgency: str = "routine"  # "immediate", "urgent", "routine", "none"
    follow_up_timeframe: Optional[str] = None
    safety_netting: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)


class WHAMProtocol:
    """
    WHAM Protocol implementation for structured medical consultations.
    
    Provides systematic approach to:
    - Patient assessment (Who)
    - Symptom evaluation (How)
    - Treatment recommendations (Action)
    - Safety monitoring (Monitoring)
    """
    
    def __init__(self):
        self.snomed_loader = SNOMEDLoader()
        self.red_flag_detector = RedFlagDetector()
        self.current_stage = WHAMStage.WHO
        self.assessment = WHAMAssessment(
            patient_context=PatientContext(),
            symptom_presentation=SymptomPresentation()
        )
    
    def process_patient_input(self, user_input: str, stage: Optional[WHAMStage] = None) -> Dict[str, Any]:
        """
        Process patient input according to WHAM protocol.
        
        Args:
            user_input: Patient's natural language input
            stage: Current WHAM stage (auto-detected if None)
            
        Returns:
            Response with next questions and extracted information
        """
        if stage:
            self.current_stage = stage
        
        # Extract information based on current stage
        if self.current_stage == WHAMStage.WHO:
            return self._process_who_stage(user_input)
        elif self.current_stage == WHAMStage.HOW:
            return self._process_how_stage(user_input)
        elif self.current_stage == WHAMStage.ACTION:
            return self._process_action_stage(user_input)
        elif self.current_stage == WHAMStage.MONITORING:
            return self._process_monitoring_stage(user_input)
    
    def _process_who_stage(self, user_input: str) -> Dict[str, Any]:
        """Process WHO stage - patient demographics and context."""
        # Extract age
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|yo|y\.?o\.?)\b', user_input, re.IGNORECASE)
        if age_match:
            self.assessment.patient_context.age = int(age_match.group(1))
        
        # Extract gender
        if re.search(r'\b(?:female|woman|girl|she|her)\b', user_input, re.IGNORECASE):
            self.assessment.patient_context.gender = "female"
        elif re.search(r'\b(?:male|man|boy|he|him)\b', user_input, re.IGNORECASE):
            self.assessment.patient_context.gender = "male"
        
        # Extract pregnancy status
        if re.search(r'\b(?:pregnant|pregnancy|expecting)\b', user_input, re.IGNORECASE):
            self.assessment.patient_context.pregnancy_status = True
        
        # Extract medications
        med_patterns = [
            r'taking\s+([^.]+?)(?:\.|$)',
            r'on\s+([^.]+?)(?:\.|$)',
            r'medication[s]?\s*:?\s*([^.]+?)(?:\.|$)'
        ]
        for pattern in med_patterns:
            matches = re.findall(pattern, user_input, re.IGNORECASE)
            for match in matches:
                medications = [med.strip() for med in match.split(',')]
                self.assessment.patient_context.current_medications.extend(medications)
        
        # Generate next questions
        next_questions = self._generate_who_questions()
        
        # Check if ready to move to HOW stage
        if self._is_who_stage_complete():
            self.current_stage = WHAMStage.HOW
            next_questions.extend(self._generate_how_questions())
        
        return {
            "stage": "WHO",
            "extracted_info": {
                "age": self.assessment.patient_context.age,
                "gender": self.assessment.patient_context.gender,
                "medications": self.assessment.patient_context.current_medications
            },
            "next_questions": next_questions,
            "stage_complete": self._is_who_stage_complete()
        }
    
    def _process_how_stage(self, user_input: str) -> Dict[str, Any]:
        """Process HOW stage - symptom presentation."""
        # Extract primary complaint
        if not self.assessment.symptom_presentation.primary_complaint:
            self.assessment.symptom_presentation.primary_complaint = user_input
        
        # Extract duration
        duration_patterns = [
            r'(?:for|since|about|around)\s+(\d+)\s*(day|week|month|year)s?',
            r'(\d+)\s*(day|week|month|year)s?\s*(?:ago|now)',
            r'started\s+(\d+)\s*(day|week|month|year)s?\s*ago'
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                self.assessment.symptom_presentation.duration = f"{match.group(1)} {match.group(2)}s"
                break
        
        # Extract severity (1-10 scale)
        severity_match = re.search(r'(?:pain|severity|intense|bad)\s*(?:is|of|about)?\s*(\d{1,2})(?:/10|out of 10)?', user_input, re.IGNORECASE)
        if severity_match:
            severity = int(severity_match.group(1))
            if 1 <= severity <= 10:
                self.assessment.symptom_presentation.severity = severity
        
        # Extract location
        body_parts = ['head', 'chest', 'abdomen', 'back', 'leg', 'arm', 'throat', 'stomach', 'knee', 'shoulder']
        for part in body_parts:
            if re.search(rf'\b{part}\b', user_input, re.IGNORECASE):
                self.assessment.symptom_presentation.location = part
                break
        
        # Check for red flags
        red_flags = self.red_flag_detector.detect_red_flags(user_input, self.assessment.patient_context)
        self.assessment.red_flags.extend(red_flags)
        
        # Generate next questions
        next_questions = self._generate_how_questions()
        
        # Check if ready for ACTION stage
        if self._is_how_stage_complete():
            self.current_stage = WHAMStage.ACTION
            self._generate_recommendations()
        
        return {
            "stage": "HOW",
            "extracted_info": {
                "primary_complaint": self.assessment.symptom_presentation.primary_complaint,
                "duration": self.assessment.symptom_presentation.duration,
                "severity": self.assessment.symptom_presentation.severity,
                "location": self.assessment.symptom_presentation.location
            },
            "red_flags": red_flags,
            "next_questions": next_questions,
            "stage_complete": self._is_how_stage_complete()
        }
    
    def _process_action_stage(self, user_input: str) -> Dict[str, Any]:
        """Process ACTION stage - treatment recommendations."""
        # Generate recommendations if not already done
        if not self.assessment.recommendations:
            self._generate_recommendations()
        
        self.current_stage = WHAMStage.MONITORING
        
        return {
            "stage": "ACTION",
            "recommendations": [
                {
                    "category": rec.category,
                    "recommendation": rec.recommendation,
                    "rationale": rec.rationale,
                    "safety_considerations": rec.safety_considerations
                }
                for rec in self.assessment.recommendations
            ],
            "referral_urgency": self.assessment.referral_urgency,
            "next_questions": self._generate_monitoring_questions()
        }
    
    def _process_monitoring_stage(self, user_input: str) -> Dict[str, Any]:
        """Process MONITORING stage - follow-up and safety."""
        # Generate safety netting advice
        safety_netting = self._generate_safety_netting()
        self.assessment.safety_netting.extend(safety_netting)
        
        return {
            "stage": "MONITORING",
            "safety_netting": safety_netting,
            "follow_up_timeframe": self.assessment.follow_up_timeframe,
            "assessment_complete": True
        }
    
    def _generate_who_questions(self) -> List[str]:
        """Generate WHO stage questions."""
        questions = []
        
        if not self.assessment.patient_context.age:
            questions.append("How old are you?")
        
        if not self.assessment.patient_context.gender:
            questions.append("Are you male or female?")
        
        if (self.assessment.patient_context.gender == "female" and 
            self.assessment.patient_context.age and 
            15 <= self.assessment.patient_context.age <= 50 and
            self.assessment.patient_context.pregnancy_status is None):
            questions.append("Are you currently pregnant or could you be pregnant?")
        
        if not self.assessment.patient_context.current_medications:
            questions.append("Are you currently taking any medications, including over-the-counter medicines?")
        
        return questions
    
    def _generate_how_questions(self) -> List[str]:
        """Generate HOW stage questions."""
        questions = []
        
        if not self.assessment.symptom_presentation.duration:
            questions.append("How long have you had this problem?")
        
        if self.assessment.symptom_presentation.severity is None:
            questions.append("On a scale of 1-10, how severe is your discomfort?")
        
        if not self.assessment.symptom_presentation.location:
            questions.append("Where exactly do you feel the problem?")
        
        if not self.assessment.symptom_presentation.aggravating_factors:
            questions.append("What makes it worse?")
        
        if not self.assessment.symptom_presentation.relieving_factors:
            questions.append("What makes it better?")
        
        return questions
    
    def _generate_monitoring_questions(self) -> List[str]:
        """Generate MONITORING stage questions."""
        return [
            "Do you understand when to seek further medical attention?",
            "Are you comfortable with the recommended treatment plan?",
            "Do you have any questions about the safety advice?"
        ]
    
    def _is_who_stage_complete(self) -> bool:
        """Check if WHO stage is complete."""
        context = self.assessment.patient_context
        return (context.age is not None and 
                context.gender is not None and
                (context.gender != "female" or context.pregnancy_status is not None))
    
    def _is_how_stage_complete(self) -> bool:
        """Check if HOW stage is complete."""
        symptoms = self.assessment.symptom_presentation
        return (symptoms.primary_complaint and 
                symptoms.duration is not None and
                symptoms.severity is not None)
    
    def _generate_recommendations(self):
        """Generate treatment recommendations based on assessment."""
        # This would integrate with clinical decision support systems
        # For now, provide basic recommendations
        
        if self.assessment.red_flags:
            # Immediate referral for red flags
            self.assessment.recommendations.append(WHAMRecommendation(
                category="referral",
                recommendation="Seek immediate medical attention",
                rationale="Red flag symptoms detected requiring urgent evaluation",
                safety_considerations=["Do not delay seeking medical care"],
                monitoring_required=True
            ))
            self.assessment.referral_urgency = "immediate"
        else:
            # Basic self-care recommendations
            self.assessment.recommendations.append(WHAMRecommendation(
                category="self_care",
                recommendation="Rest and monitor symptoms",
                rationale="Symptoms appear manageable with conservative treatment",
                safety_considerations=["Return if symptoms worsen"],
                monitoring_required=True
            ))
            self.assessment.referral_urgency = "routine"
    
    def _generate_safety_netting(self) -> List[str]:
        """Generate safety netting advice."""
        safety_advice = [
            "Return for medical attention if symptoms worsen significantly",
            "Seek immediate help if you develop severe pain, difficulty breathing, or other concerning symptoms",
            "Follow up with your healthcare provider if symptoms don't improve as expected"
        ]
        
        # Add specific advice based on symptoms and patient context
        if self.assessment.patient_context.age and self.assessment.patient_context.age > 65:
            safety_advice.append("Older adults should have a lower threshold for seeking medical attention")
        
        if self.assessment.patient_context.pregnancy_status:
            safety_advice.append("Pregnant women should consult with their healthcare provider before taking any medications")
        
        return safety_advice
    
    def get_complete_assessment(self) -> Dict[str, Any]:
        """Get the complete WHAM assessment."""
        return {
            "patient_context": {
                "age": self.assessment.patient_context.age,
                "gender": self.assessment.patient_context.gender,
                "pregnancy_status": self.assessment.patient_context.pregnancy_status,
                "medications": self.assessment.patient_context.current_medications,
                "allergies": self.assessment.patient_context.allergies
            },
            "symptom_presentation": {
                "primary_complaint": self.assessment.symptom_presentation.primary_complaint,
                "duration": self.assessment.symptom_presentation.duration,
                "severity": self.assessment.symptom_presentation.severity,
                "location": self.assessment.symptom_presentation.location
            },
            "recommendations": [
                {
                    "category": rec.category,
                    "recommendation": rec.recommendation,
                    "rationale": rec.rationale,
                    "safety_considerations": rec.safety_considerations
                }
                for rec in self.assessment.recommendations
            ],
            "red_flags": self.assessment.red_flags,
            "referral_urgency": self.assessment.referral_urgency,
            "safety_netting": self.assessment.safety_netting
        }
