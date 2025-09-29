"""
Red Flag Detection System for Medical Safety

Identifies critical symptoms requiring immediate medical attention.
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ...utils.logging.logger import get_logger

logger = get_logger(__name__)


class RedFlagSeverity(Enum):
    """Red flag severity levels."""
    CRITICAL = "critical"  # Immediate emergency care
    HIGH = "high"         # Urgent medical attention
    MODERATE = "moderate" # Same-day medical review


@dataclass
class RedFlag:
    """Red flag detection result."""
    flag_type: str
    description: str
    severity: RedFlagSeverity
    keywords_matched: List[str]
    recommendation: str


class RedFlagDetector:
    """
    Comprehensive red flag detection system for medical safety.
    
    Identifies symptoms requiring immediate or urgent medical attention
    across multiple body systems and conditions.
    """
    
    def __init__(self):
        self.red_flag_patterns = self._initialize_red_flag_patterns()
    
    def _initialize_red_flag_patterns(self) -> Dict[str, Dict]:
        """Initialize red flag detection patterns."""
        return {
            # Cardiovascular red flags
            "chest_pain_cardiac": {
                "patterns": [
                    r'\b(?:chest|heart)\s+(?:pain|ache|pressure|tightness|crushing|squeezing)\b',
                    r'\bchest\s+feels?\s+(?:tight|heavy|compressed)\b',
                    r'\bheart\s+(?:attack|feels?\s+like\s+exploding)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Chest pain suggestive of cardiac event",
                "recommendation": "Call emergency services immediately (999/911)"
            },
            
            "shortness_of_breath_severe": {
                "patterns": [
                    r'\b(?:can\'?t|cannot|unable\s+to)\s+(?:breathe|catch\s+breath)\b',
                    r'\bsevere\s+(?:shortness\s+of\s+breath|breathlessness)\b',
                    r'\bgasping\s+for\s+(?:air|breath)\b',
                    r'\bfeels?\s+like\s+(?:drowning|suffocating)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Severe respiratory distress",
                "recommendation": "Call emergency services immediately"
            },
            
            # Neurological red flags
            "stroke_symptoms": {
                "patterns": [
                    r'\b(?:face|facial)\s+(?:drooping|weakness|numbness)\b',
                    r'\b(?:arm|leg)\s+(?:weakness|paralysis|can\'?t\s+move)\b',
                    r'\bslurred\s+speech\b',
                    r'\bsudden\s+(?:confusion|dizziness|loss\s+of\s+balance)\b',
                    r'\bsudden\s+severe\s+headache\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Possible stroke symptoms",
                "recommendation": "Call emergency services immediately - time critical"
            },
            
            "severe_headache": {
                "patterns": [
                    r'\bworst\s+headache\s+(?:of\s+my\s+life|ever)\b',
                    r'\bthundeclap\s+headache\b',
                    r'\bsudden\s+onset\s+severe\s+headache\b',
                    r'\bheadache\s+with\s+(?:neck\s+stiffness|fever|rash)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Severe headache with concerning features",
                "recommendation": "Seek immediate emergency care"
            },
            
            # Abdominal red flags
            "severe_abdominal_pain": {
                "patterns": [
                    r'\bsevere\s+(?:abdominal|stomach|belly)\s+pain\b',
                    r'\b(?:abdominal|stomach)\s+pain\s+(?:10/10|unbearable|excruciating)\b',
                    r'\bappendix\s+(?:pain|burst|rupture)\b',
                    r'\bboard-like\s+(?:abdomen|stomach)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Severe abdominal pain requiring urgent evaluation",
                "recommendation": "Seek immediate medical attention"
            },
            
            # Bleeding red flags
            "severe_bleeding": {
                "patterns": [
                    r'\b(?:heavy|severe|uncontrolled|massive)\s+bleeding\b',
                    r'\bbleeding\s+(?:won\'?t\s+stop|continuously|profusely)\b',
                    r'\b(?:vomiting|coughing\s+up|spitting)\s+blood\b',
                    r'\bblood\s+in\s+(?:urine|stool|vomit)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Severe or uncontrolled bleeding",
                "recommendation": "Apply pressure and seek immediate emergency care"
            },
            
            # Pregnancy red flags
            "pregnancy_complications": {
                "patterns": [
                    r'\bpregnant\s+(?:and|with)\s+(?:bleeding|cramping|severe\s+pain)\b',
                    r'\bmiscarriage\b',
                    r'\bectopic\s+pregnancy\b',
                    r'\bpreeclampsia\b',
                    r'\bsevere\s+(?:morning\s+sickness|nausea\s+and\s+vomiting)\b'
                ],
                "severity": RedFlagSeverity.HIGH,
                "description": "Pregnancy-related complications",
                "recommendation": "Contact maternity services or seek urgent medical care"
            },
            
            # Infection red flags
            "sepsis_signs": {
                "patterns": [
                    r'\b(?:high\s+fever|temperature\s+(?:over\s+)?(?:39|40|101|102))\b',
                    r'\bfever\s+with\s+(?:rash|confusion|difficulty\s+breathing)\b',
                    r'\bsepsis\b',
                    r'\bfeels?\s+(?:very\s+unwell|like\s+dying)\b'
                ],
                "severity": RedFlagSeverity.HIGH,
                "description": "Signs suggestive of serious infection",
                "recommendation": "Seek urgent medical attention"
            },
            
            # Mental health red flags
            "suicide_risk": {
                "patterns": [
                    r'\b(?:want\s+to\s+die|suicide|kill\s+myself|end\s+it\s+all)\b',
                    r'\b(?:no\s+point|nothing\s+to\s+live\s+for|better\s+off\s+dead)\b',
                    r'\bthoughts?\s+of\s+(?:suicide|self-harm|hurting\s+myself)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Suicide risk identified",
                "recommendation": "Contact crisis helpline or emergency services immediately"
            },
            
            # Pediatric red flags
            "pediatric_emergency": {
                "patterns": [
                    r'\b(?:baby|infant|child)\s+(?:not\s+breathing|blue|unconscious)\b',
                    r'\b(?:baby|child)\s+(?:won\'?t\s+wake\s+up|unresponsive)\b',
                    r'\bfebrile\s+(?:seizure|convulsion)\b',
                    r'\bchild\s+(?:very\s+lethargic|floppy)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Pediatric emergency",
                "recommendation": "Call emergency services immediately"
            },
            
            # Allergic reaction red flags
            "anaphylaxis": {
                "patterns": [
                    r'\b(?:severe\s+)?allergic\s+reaction\b',
                    r'\banaphylaxis\b',
                    r'\b(?:face|lips|tongue|throat)\s+swelling\b',
                    r'\b(?:difficulty\s+swallowing|throat\s+closing)\b',
                    r'\b(?:hives|rash)\s+(?:all\s+over|spreading)\b'
                ],
                "severity": RedFlagSeverity.CRITICAL,
                "description": "Severe allergic reaction",
                "recommendation": "Use EpiPen if available and call emergency services"
            },
            
            # Eye red flags
            "vision_loss": {
                "patterns": [
                    r'\bsudden\s+(?:vision\s+loss|blindness|can\'?t\s+see)\b',
                    r'\b(?:flashing\s+lights|curtain\s+across\s+vision)\b',
                    r'\bretinal\s+detachment\b',
                    r'\beye\s+(?:injury|trauma|chemical\s+in)\b'
                ],
                "severity": RedFlagSeverity.HIGH,
                "description": "Acute vision problems",
                "recommendation": "Seek urgent ophthalmology or emergency care"
            }
        }
    
    def detect_red_flags(self, text: str, patient_context: Optional[Any] = None) -> List[RedFlag]:
        """
        Detect red flags in patient text.
        
        Args:
            text: Patient's description of symptoms
            patient_context: Additional patient context (age, gender, etc.)
            
        Returns:
            List of detected red flags
        """
        detected_flags = []
        text_lower = text.lower()
        
        for flag_type, flag_config in self.red_flag_patterns.items():
            matched_keywords = []
            
            for pattern in flag_config["patterns"]:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    matched_keywords.extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
            
            if matched_keywords:
                red_flag = RedFlag(
                    flag_type=flag_type,
                    description=flag_config["description"],
                    severity=flag_config["severity"],
                    keywords_matched=list(set(matched_keywords)),
                    recommendation=flag_config["recommendation"]
                )
                detected_flags.append(red_flag)
        
        # Apply context-specific logic
        if patient_context:
            detected_flags = self._apply_context_filters(detected_flags, patient_context)
        
        # Sort by severity (critical first)
        detected_flags.sort(key=lambda x: (
            0 if x.severity == RedFlagSeverity.CRITICAL else
            1 if x.severity == RedFlagSeverity.HIGH else 2
        ))
        
        return detected_flags
    
    def _apply_context_filters(self, flags: List[RedFlag], patient_context: Any) -> List[RedFlag]:
        """Apply patient context to refine red flag detection."""
        filtered_flags = []
        
        for flag in flags:
            # Age-specific filtering
            if hasattr(patient_context, 'age') and patient_context.age:
                if flag.flag_type == "pediatric_emergency" and patient_context.age > 16:
                    continue  # Skip pediatric flags for adults
                
                if flag.flag_type == "pregnancy_complications":
                    if (hasattr(patient_context, 'gender') and 
                        patient_context.gender != "female"):
                        continue  # Skip pregnancy flags for males
                    
                    if patient_context.age < 12 or patient_context.age > 55:
                        continue  # Skip for unlikely pregnancy ages
            
            filtered_flags.append(flag)
        
        return filtered_flags
    
    def get_emergency_response(self, red_flags: List[RedFlag]) -> Dict[str, Any]:
        """
        Generate emergency response recommendations based on detected red flags.
        
        Args:
            red_flags: List of detected red flags
            
        Returns:
            Emergency response recommendations
        """
        if not red_flags:
            return {
                "emergency_level": "none",
                "action_required": "routine_care",
                "message": "No immediate red flags detected"
            }
        
        # Check for critical flags
        critical_flags = [f for f in red_flags if f.severity == RedFlagSeverity.CRITICAL]
        if critical_flags:
            return {
                "emergency_level": "critical",
                "action_required": "immediate_emergency_care",
                "message": "CRITICAL: Call emergency services immediately (999/911)",
                "flags": [f.description for f in critical_flags],
                "recommendations": [f.recommendation for f in critical_flags]
            }
        
        # Check for high severity flags
        high_flags = [f for f in red_flags if f.severity == RedFlagSeverity.HIGH]
        if high_flags:
            return {
                "emergency_level": "high",
                "action_required": "urgent_medical_attention",
                "message": "URGENT: Seek immediate medical attention",
                "flags": [f.description for f in high_flags],
                "recommendations": [f.recommendation for f in high_flags]
            }
        
        # Moderate flags
        return {
            "emergency_level": "moderate",
            "action_required": "same_day_medical_review",
            "message": "Seek medical attention today",
            "flags": [f.description for f in red_flags],
            "recommendations": [f.recommendation for f in red_flags]
        }
    
    def add_custom_red_flag(self, flag_type: str, patterns: List[str], 
                           severity: RedFlagSeverity, description: str, 
                           recommendation: str):
        """Add custom red flag pattern."""
        self.red_flag_patterns[flag_type] = {
            "patterns": patterns,
            "severity": severity,
            "description": description,
            "recommendation": recommendation
        }
        
        logger.info(f"Added custom red flag: {flag_type}")
    
    def get_red_flag_summary(self) -> Dict[str, int]:
        """Get summary of available red flag patterns."""
        summary = {}
        for flag_type, config in self.red_flag_patterns.items():
            severity = config["severity"].value
            summary[severity] = summary.get(severity, 0) + 1
        
        return summary
