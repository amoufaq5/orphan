"""
AS METHOD Protocol Implementation

AS METHOD: Age, Self-care, Medication, Extra info, Time, History, Other symptoms, Danger
Comprehensive patient assessment framework for medical consultations.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from ...utils.logging.logger import get_logger
from ..safety.red_flags import RedFlagDetector

logger = get_logger(__name__)


class ASMethodStage(Enum):
    """AS METHOD protocol stages."""
    AGE = "age"
    SELF_CARE = "self_care"
    MEDICATION = "medication"
    EXTRA_INFO = "extra_info"
    TIME = "time"
    HISTORY = "history"
    OTHER_SYMPTOMS = "other_symptoms"
    DANGER = "danger"


@dataclass
class ASMethodAssessment:
    """Complete AS METHOD assessment."""
    age: Optional[int] = None
    self_care_attempted: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)
    symptom_timeline: Optional[str] = None
    medical_history: List[str] = field(default_factory=list)
    other_symptoms: List[str] = field(default_factory=list)
    danger_signs: List[str] = field(default_factory=list)
    assessment_complete: bool = False


class ASMethodProtocol:
    """
    AS METHOD Protocol implementation for systematic patient assessment.
    
    Provides structured approach covering:
    - Age considerations
    - Self-care measures tried
    - Current medications
    - Extra patient information
    - Timeline of symptoms
    - Medical history
    - Other associated symptoms
    - Danger signs assessment
    """
    
    def __init__(self):
        self.red_flag_detector = RedFlagDetector()
        self.current_stage = ASMethodStage.AGE
        self.assessment = ASMethodAssessment()
    
    def process_patient_input(self, user_input: str, stage: Optional[ASMethodStage] = None) -> Dict[str, Any]:
        """Process patient input according to AS METHOD protocol."""
        if stage:
            self.current_stage = stage
        
        # Process based on current stage
        stage_processors = {
            ASMethodStage.AGE: self._process_age,
            ASMethodStage.SELF_CARE: self._process_self_care,
            ASMethodStage.MEDICATION: self._process_medication,
            ASMethodStage.EXTRA_INFO: self._process_extra_info,
            ASMethodStage.TIME: self._process_time,
            ASMethodStage.HISTORY: self._process_history,
            ASMethodStage.OTHER_SYMPTOMS: self._process_other_symptoms,
            ASMethodStage.DANGER: self._process_danger
        }
        
        return stage_processors[self.current_stage](user_input)
    
    def _process_age(self, user_input: str) -> Dict[str, Any]:
        """Process Age stage."""
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|yo|y\.?o\.?)\b', user_input, re.IGNORECASE)
        if age_match:
            self.assessment.age = int(age_match.group(1))
        
        if self.assessment.age:
            self.current_stage = ASMethodStage.SELF_CARE
        
        return {
            "stage": "AGE",
            "extracted_info": {"age": self.assessment.age},
            "next_questions": ["What have you tried to treat this problem yourself?"] if self.assessment.age else ["How old are you?"],
            "stage_complete": self.assessment.age is not None
        }
    
    def _process_self_care(self, user_input: str) -> Dict[str, Any]:
        """Process Self-care stage."""
        # Extract self-care measures
        self_care_keywords = ['rest', 'ice', 'heat', 'massage', 'exercise', 'stretching', 'water', 'tea', 'honey']
        for keyword in self_care_keywords:
            if re.search(rf'\b{keyword}\b', user_input, re.IGNORECASE):
                if keyword not in self.assessment.self_care_attempted:
                    self.assessment.self_care_attempted.append(keyword)
        
        self.current_stage = ASMethodStage.MEDICATION
        
        return {
            "stage": "SELF_CARE",
            "extracted_info": {"self_care": self.assessment.self_care_attempted},
            "next_questions": ["Are you currently taking any medications?"],
            "stage_complete": True
        }
    
    def _process_medication(self, user_input: str) -> Dict[str, Any]:
        """Process Medication stage."""
        # Extract medications
        if not re.search(r'\b(?:no|none|nothing)\b', user_input, re.IGNORECASE):
            medications = [med.strip() for med in user_input.split(',')]
            self.assessment.current_medications.extend(medications)
        
        self.current_stage = ASMethodStage.EXTRA_INFO
        
        return {
            "stage": "MEDICATION", 
            "extracted_info": {"medications": self.assessment.current_medications},
            "next_questions": ["Is there anything else about your health I should know?"],
            "stage_complete": True
        }
    
    def _process_extra_info(self, user_input: str) -> Dict[str, Any]:
        """Process Extra information stage."""
        self.assessment.extra_info["additional_context"] = user_input
        self.current_stage = ASMethodStage.TIME
        
        return {
            "stage": "EXTRA_INFO",
            "extracted_info": {"extra_info": self.assessment.extra_info},
            "next_questions": ["When did this problem start?"],
            "stage_complete": True
        }
    
    def _process_time(self, user_input: str) -> Dict[str, Any]:
        """Process Time/Timeline stage."""
        # Extract timeline
        timeline_patterns = [
            r'(?:started|began)\s+(\d+)\s*(day|week|month|year)s?\s*ago',
            r'(?:for|since)\s+(\d+)\s*(day|week|month|year)s?',
            r'(\d+)\s*(day|week|month|year)s?\s*(?:ago|now)'
        ]
        
        for pattern in timeline_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                self.assessment.symptom_timeline = f"{match.group(1)} {match.group(2)}s ago"
                break
        
        self.current_stage = ASMethodStage.HISTORY
        
        return {
            "stage": "TIME",
            "extracted_info": {"timeline": self.assessment.symptom_timeline},
            "next_questions": ["Do you have any relevant medical history?"],
            "stage_complete": True
        }
    
    def _process_history(self, user_input: str) -> Dict[str, Any]:
        """Process History stage."""
        if not re.search(r'\b(?:no|none|nothing)\b', user_input, re.IGNORECASE):
            history_items = [item.strip() for item in user_input.split(',')]
            self.assessment.medical_history.extend(history_items)
        
        self.current_stage = ASMethodStage.OTHER_SYMPTOMS
        
        return {
            "stage": "HISTORY",
            "extracted_info": {"history": self.assessment.medical_history},
            "next_questions": ["Are you experiencing any other symptoms?"],
            "stage_complete": True
        }
    
    def _process_other_symptoms(self, user_input: str) -> Dict[str, Any]:
        """Process Other symptoms stage."""
        if not re.search(r'\b(?:no|none|nothing)\b', user_input, re.IGNORECASE):
            symptoms = [symptom.strip() for symptom in user_input.split(',')]
            self.assessment.other_symptoms.extend(symptoms)
        
        self.current_stage = ASMethodStage.DANGER
        
        return {
            "stage": "OTHER_SYMPTOMS",
            "extracted_info": {"other_symptoms": self.assessment.other_symptoms},
            "next_questions": ["Have you experienced any severe or concerning symptoms?"],
            "stage_complete": True
        }
    
    def _process_danger(self, user_input: str) -> Dict[str, Any]:
        """Process Danger signs stage."""
        # Detect danger signs
        danger_signs = self.red_flag_detector.detect_red_flags(user_input, None)
        self.assessment.danger_signs.extend(danger_signs)
        
        self.assessment.assessment_complete = True
        
        return {
            "stage": "DANGER",
            "extracted_info": {"danger_signs": self.assessment.danger_signs},
            "next_questions": [],
            "stage_complete": True,
            "assessment_complete": True
        }
    
    def get_complete_assessment(self) -> Dict[str, Any]:
        """Get complete AS METHOD assessment."""
        return {
            "age": self.assessment.age,
            "self_care_attempted": self.assessment.self_care_attempted,
            "current_medications": self.assessment.current_medications,
            "extra_info": self.assessment.extra_info,
            "symptom_timeline": self.assessment.symptom_timeline,
            "medical_history": self.assessment.medical_history,
            "other_symptoms": self.assessment.other_symptoms,
            "danger_signs": self.assessment.danger_signs,
            "assessment_complete": self.assessment.assessment_complete
        }
