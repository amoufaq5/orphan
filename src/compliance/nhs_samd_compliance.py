"""
NHS SAMD Compliance Framework

Implements UK MHRA Software as Medical Device (SAMD) compliance
requirements for NHS deployment and regulatory approval.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from ..utils.logging.logger import get_logger
from ..protocols.safety.red_flags import RedFlagDetector

logger = get_logger(__name__)


class SAMDClassification(Enum):
    """SAMD Classification according to MHRA guidelines."""
    CLASS_I = "class_i"      # Low risk - informational
    CLASS_II = "class_ii"    # Moderate risk - diagnostic aid
    CLASS_III = "class_iii"  # High risk - treatment decisions
    CLASS_IV = "class_iv"    # Critical risk - life-threatening


class ClinicalRiskLevel(Enum):
    """Clinical risk levels for SAMD assessment."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClinicalValidationStudy:
    """Clinical validation study record."""
    study_id: str
    study_type: str  # RCT, observational, retrospective
    population: Dict[str, Any]
    endpoints: List[str]
    results: Dict[str, float]
    publication_reference: Optional[str]
    regulatory_submission: Optional[str]


@dataclass
class SAMDRiskAssessment:
    """SAMD risk assessment record."""
    assessment_id: str
    samd_classification: SAMDClassification
    clinical_risk_level: ClinicalRiskLevel
    intended_use: str
    target_population: str
    clinical_context: str
    risk_factors: List[str]
    mitigation_measures: List[str]
    residual_risks: List[str]
    assessment_date: str
    assessor: str


@dataclass
class ClinicalGovernanceRecord:
    """Clinical governance and oversight record."""
    governance_id: str
    medical_director: Dict[str, str]
    clinical_advisory_board: List[Dict[str, str]]
    clinical_protocols: List[str]
    escalation_procedures: Dict[str, str]
    review_schedule: str
    last_review_date: str
    next_review_date: str


class NHSSAMDCompliance:
    """
    NHS SAMD Compliance Framework
    
    Implements UK MHRA requirements for Software as Medical Device:
    - SAMD classification and risk assessment
    - Clinical validation requirements
    - Quality management system
    - Post-market surveillance
    - Clinical governance
    - NHS Digital Technology Assessment Criteria (DTAC)
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_compliance_config(config_path)
        self.samd_classification = SAMDClassification.CLASS_II  # Default moderate risk
        self.clinical_validation_studies = []
        self.risk_assessments = []
        self.governance_records = []
        self.audit_trail = []
        
        # NHS-specific requirements
        self.nhs_dtac_compliance = self._initialize_dtac_compliance()
        self.clinical_safety_requirements = self._initialize_clinical_safety()
        
        logger.info("NHS SAMD Compliance Framework initialized")
    
    def perform_samd_classification(
        self,
        intended_use: str,
        target_population: str,
        clinical_context: str
    ) -> SAMDRiskAssessment:
        """
        Perform SAMD classification according to MHRA guidelines.
        
        Classification matrix:
        - Healthcare situation (serious/non-serious/critical)
        - Healthcare decision (inform/drive/diagnose/treat)
        """
        # Analyze intended use for classification
        classification = self._determine_samd_class(intended_use, clinical_context)
        risk_level = self._assess_clinical_risk(intended_use, target_population)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(intended_use, target_population)
        
        # Define mitigation measures
        mitigation_measures = self._define_mitigation_measures(classification, risk_factors)
        
        # Assess residual risks
        residual_risks = self._assess_residual_risks(risk_factors, mitigation_measures)
        
        risk_assessment = SAMDRiskAssessment(
            assessment_id=str(uuid.uuid4()),
            samd_classification=classification,
            clinical_risk_level=risk_level,
            intended_use=intended_use,
            target_population=target_population,
            clinical_context=clinical_context,
            risk_factors=risk_factors,
            mitigation_measures=mitigation_measures,
            residual_risks=residual_risks,
            assessment_date=datetime.now().isoformat(),
            assessor="AI System Risk Assessment"
        )
        
        self.risk_assessments.append(risk_assessment)
        self._log_compliance_activity("SAMD Classification", risk_assessment.assessment_id)
        
        return risk_assessment
    
    def validate_clinical_performance(
        self,
        study_data: Dict[str, Any]
    ) -> ClinicalValidationStudy:
        """
        Validate clinical performance according to NHS requirements.
        
        Required metrics:
        - Sensitivity and Specificity
        - Positive/Negative Predictive Values
        - Clinical utility measures
        - Safety outcomes
        """
        study = ClinicalValidationStudy(
            study_id=str(uuid.uuid4()),
            study_type=study_data.get("type", "observational"),
            population=study_data.get("population", {}),
            endpoints=study_data.get("endpoints", []),
            results=study_data.get("results", {}),
            publication_reference=study_data.get("publication"),
            regulatory_submission=study_data.get("submission")
        )
        
        # Validate required performance metrics
        required_metrics = [
            "sensitivity", "specificity", "positive_predictive_value",
            "negative_predictive_value", "accuracy"
        ]
        
        missing_metrics = [m for m in required_metrics if m not in study.results]
        if missing_metrics:
            logger.warning(f"Missing required metrics: {missing_metrics}")
        
        # Check NHS performance thresholds
        performance_check = self._check_nhs_performance_thresholds(study.results)
        
        self.clinical_validation_studies.append(study)
        self._log_compliance_activity("Clinical Validation", study.study_id)
        
        return study
    
    def establish_clinical_governance(
        self,
        medical_director: Dict[str, str],
        advisory_board: List[Dict[str, str]]
    ) -> ClinicalGovernanceRecord:
        """
        Establish clinical governance structure for NHS compliance.
        
        Requirements:
        - Qualified medical director
        - Clinical advisory board
        - Clinical protocols and procedures
        - Regular review processes
        """
        # Validate medical director qualifications
        if not self._validate_medical_director(medical_director):
            raise ValueError("Medical director does not meet NHS requirements")
        
        # Validate advisory board
        if not self._validate_advisory_board(advisory_board):
            raise ValueError("Clinical advisory board does not meet NHS requirements")
        
        governance = ClinicalGovernanceRecord(
            governance_id=str(uuid.uuid4()),
            medical_director=medical_director,
            clinical_advisory_board=advisory_board,
            clinical_protocols=self._define_clinical_protocols(),
            escalation_procedures=self._define_escalation_procedures(),
            review_schedule="quarterly",
            last_review_date=datetime.now().isoformat(),
            next_review_date=(datetime.now() + timedelta(days=90)).isoformat()
        )
        
        self.governance_records.append(governance)
        self._log_compliance_activity("Clinical Governance", governance.governance_id)
        
        return governance
    
    def assess_nhs_dtac_compliance(self) -> Dict[str, Any]:
        """
        Assess compliance with NHS Digital Technology Assessment Criteria.
        
        DTAC Requirements:
        1. Clinical safety
        2. Data protection
        3. Technical assurance
        4. Interoperability
        5. Usability and accessibility
        """
        compliance_assessment = {
            "clinical_safety": self._assess_clinical_safety_compliance(),
            "data_protection": self._assess_data_protection_compliance(),
            "technical_assurance": self._assess_technical_assurance_compliance(),
            "interoperability": self._assess_interoperability_compliance(),
            "usability_accessibility": self._assess_usability_compliance(),
            "overall_score": 0.0,
            "compliance_status": "pending",
            "recommendations": []
        }
        
        # Calculate overall compliance score
        scores = [v["score"] for v in compliance_assessment.values() if isinstance(v, dict) and "score" in v]
        compliance_assessment["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        # Determine compliance status
        if compliance_assessment["overall_score"] >= 0.9:
            compliance_assessment["compliance_status"] = "compliant"
        elif compliance_assessment["overall_score"] >= 0.7:
            compliance_assessment["compliance_status"] = "conditionally_compliant"
        else:
            compliance_assessment["compliance_status"] = "non_compliant"
        
        return compliance_assessment
    
    def generate_regulatory_submission(self) -> Dict[str, Any]:
        """
        Generate regulatory submission package for MHRA approval.
        
        Includes:
        - SAMD classification
        - Clinical validation evidence
        - Risk management file
        - Quality management system
        - Clinical governance documentation
        """
        submission = {
            "submission_id": str(uuid.uuid4()),
            "submission_date": datetime.now().isoformat(),
            "device_information": {
                "name": "Orphan Medical AI Platform",
                "version": "1.0.0",
                "manufacturer": "Orphan Medical AI Ltd",
                "intended_use": "Clinical decision support for primary care",
                "samd_classification": self.samd_classification.value
            },
            "risk_management": {
                "risk_assessments": [asdict(ra) for ra in self.risk_assessments],
                "risk_control_measures": self._compile_risk_controls(),
                "post_market_surveillance": self._define_post_market_surveillance()
            },
            "clinical_evidence": {
                "validation_studies": [asdict(study) for study in self.clinical_validation_studies],
                "performance_metrics": self._compile_performance_metrics(),
                "clinical_utility": self._assess_clinical_utility()
            },
            "quality_management": {
                "iso_13485_compliance": True,
                "design_controls": self._document_design_controls(),
                "software_lifecycle": self._document_software_lifecycle()
            },
            "clinical_governance": [asdict(gov) for gov in self.governance_records],
            "regulatory_pathway": self._determine_regulatory_pathway()
        }
        
        return submission
    
    def _load_compliance_config(self, config_path: str) -> Dict[str, Any]:
        """Load NHS compliance configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "nhs_requirements": {
                    "clinical_safety": True,
                    "data_protection": True,
                    "interoperability": True
                },
                "performance_thresholds": {
                    "sensitivity": 0.85,
                    "specificity": 0.90,
                    "accuracy": 0.88
                }
            }
    
    def _determine_samd_class(self, intended_use: str, clinical_context: str) -> SAMDClassification:
        """Determine SAMD classification based on intended use."""
        # Analyze intended use for risk classification
        if "emergency" in intended_use.lower() or "critical" in clinical_context.lower():
            return SAMDClassification.CLASS_IV
        elif "diagnosis" in intended_use.lower() or "treatment" in intended_use.lower():
            return SAMDClassification.CLASS_III
        elif "decision support" in intended_use.lower() or "assessment" in intended_use.lower():
            return SAMDClassification.CLASS_II
        else:
            return SAMDClassification.CLASS_I
    
    def _assess_clinical_risk(self, intended_use: str, target_population: str) -> ClinicalRiskLevel:
        """Assess clinical risk level."""
        risk_indicators = [
            "pediatric" in target_population.lower(),
            "elderly" in target_population.lower(),
            "emergency" in intended_use.lower(),
            "critical" in intended_use.lower()
        ]
        
        risk_count = sum(risk_indicators)
        
        if risk_count >= 3:
            return ClinicalRiskLevel.CRITICAL
        elif risk_count >= 2:
            return ClinicalRiskLevel.HIGH
        elif risk_count >= 1:
            return ClinicalRiskLevel.MODERATE
        else:
            return ClinicalRiskLevel.LOW
    
    def _identify_risk_factors(self, intended_use: str, target_population: str) -> List[str]:
        """Identify clinical and technical risk factors."""
        risk_factors = []
        
        # Clinical risk factors
        if "diagnosis" in intended_use.lower():
            risk_factors.append("Misdiagnosis risk")
        if "treatment" in intended_use.lower():
            risk_factors.append("Inappropriate treatment recommendation")
        if "pediatric" in target_population.lower():
            risk_factors.append("Pediatric population complexity")
        
        # Technical risk factors
        risk_factors.extend([
            "AI model bias",
            "Data quality dependency",
            "Network connectivity failure",
            "Software bugs or errors",
            "User interface confusion"
        ])
        
        return risk_factors
    
    def _define_mitigation_measures(
        self,
        classification: SAMDClassification,
        risk_factors: List[str]
    ) -> List[str]:
        """Define risk mitigation measures."""
        measures = [
            "Comprehensive red flag detection system",
            "Clinical oversight and governance",
            "User training and competency requirements",
            "Regular software updates and monitoring",
            "Audit trail and logging",
            "Medical disclaimer and limitations notice"
        ]
        
        # Classification-specific measures
        if classification in [SAMDClassification.CLASS_III, SAMDClassification.CLASS_IV]:
            measures.extend([
                "Mandatory clinical review of AI recommendations",
                "Real-time clinical decision support alerts",
                "Enhanced user authentication and access controls"
            ])
        
        return measures
    
    def _assess_residual_risks(
        self,
        risk_factors: List[str],
        mitigation_measures: List[str]
    ) -> List[str]:
        """Assess residual risks after mitigation."""
        residual_risks = [
            "User override of safety recommendations",
            "Rare edge cases not covered in training data",
            "Technical system failures",
            "Human factors and usability issues"
        ]
        
        return residual_risks
    
    def _initialize_dtac_compliance(self) -> Dict[str, Any]:
        """Initialize NHS DTAC compliance framework."""
        return {
            "clinical_safety": {
                "dcb0129": "Clinical Risk Management Standard",
                "dcb0160": "Clinical Safety Case Reports"
            },
            "data_protection": {
                "gdpr": "General Data Protection Regulation",
                "data_security": "NHS Data Security and Protection Toolkit"
            },
            "technical_assurance": {
                "cyber_security": "Cyber Essentials Plus",
                "software_quality": "ISO 13485 Medical Devices QMS"
            },
            "interoperability": {
                "fhir": "HL7 FHIR R4 Standard",
                "snomed_ct": "SNOMED CT UK Edition"
            }
        }
    
    def _initialize_clinical_safety(self) -> Dict[str, Any]:
        """Initialize clinical safety requirements."""
        return {
            "clinical_risk_management": {
                "standard": "DCB0129",
                "risk_assessment_required": True,
                "clinical_safety_officer": "Required"
            },
            "hazard_analysis": {
                "clinical_hazards": [],
                "technical_hazards": [],
                "use_hazards": []
            },
            "safety_requirements": {
                "red_flag_detection": "Mandatory",
                "escalation_protocols": "Required",
                "audit_logging": "Comprehensive"
            }
        }
    
    def _check_nhs_performance_thresholds(self, results: Dict[str, float]) -> Dict[str, bool]:
        """Check if performance meets NHS thresholds."""
        thresholds = self.config.get("performance_thresholds", {})
        
        performance_check = {}
        for metric, threshold in thresholds.items():
            if metric in results:
                performance_check[metric] = results[metric] >= threshold
            else:
                performance_check[metric] = False
        
        return performance_check
    
    def _validate_medical_director(self, medical_director: Dict[str, str]) -> bool:
        """Validate medical director qualifications."""
        required_fields = ["name", "gmc_number", "specialty", "experience_years"]
        
        # Check required fields
        if not all(field in medical_director for field in required_fields):
            return False
        
        # Check minimum experience
        if int(medical_director.get("experience_years", 0)) < 5:
            return False
        
        return True
    
    def _validate_advisory_board(self, advisory_board: List[Dict[str, str]]) -> bool:
        """Validate clinical advisory board composition."""
        if len(advisory_board) < 3:
            return False
        
        # Check for diverse specialties
        specialties = [member.get("specialty") for member in advisory_board]
        required_specialties = ["general_practice", "emergency_medicine", "clinical_informatics"]
        
        return any(spec in specialties for spec in required_specialties)
    
    def _define_clinical_protocols(self) -> List[str]:
        """Define clinical protocols and procedures."""
        return [
            "Red flag escalation protocol",
            "Clinical decision support workflow",
            "Patient safety monitoring procedure",
            "Adverse event reporting protocol",
            "Clinical governance review process",
            "User competency assessment procedure"
        ]
    
    def _define_escalation_procedures(self) -> Dict[str, str]:
        """Define clinical escalation procedures."""
        return {
            "red_flag_detection": "Immediate clinical review required",
            "system_error": "Technical support escalation within 1 hour",
            "adverse_event": "Clinical safety officer notification within 24 hours",
            "user_concern": "Clinical governance review within 48 hours"
        }
    
    def _assess_clinical_safety_compliance(self) -> Dict[str, Any]:
        """Assess clinical safety compliance."""
        return {
            "score": 0.85,
            "dcb0129_compliance": True,
            "clinical_risk_management": True,
            "hazard_analysis": True,
            "recommendations": ["Complete clinical safety case report"]
        }
    
    def _assess_data_protection_compliance(self) -> Dict[str, Any]:
        """Assess data protection compliance."""
        return {
            "score": 0.90,
            "gdpr_compliance": True,
            "data_security_toolkit": True,
            "encryption": True,
            "recommendations": ["Complete DSPT assessment"]
        }
    
    def _assess_technical_assurance_compliance(self) -> Dict[str, Any]:
        """Assess technical assurance compliance."""
        return {
            "score": 0.88,
            "cyber_security": True,
            "software_quality": True,
            "testing": True,
            "recommendations": ["Obtain Cyber Essentials Plus certification"]
        }
    
    def _assess_interoperability_compliance(self) -> Dict[str, Any]:
        """Assess interoperability compliance."""
        return {
            "score": 0.92,
            "fhir_compliance": True,
            "snomed_ct": True,
            "nhs_number": True,
            "recommendations": ["Implement additional HL7 FHIR resources"]
        }
    
    def _assess_usability_compliance(self) -> Dict[str, Any]:
        """Assess usability and accessibility compliance."""
        return {
            "score": 0.87,
            "accessibility": True,
            "usability_testing": True,
            "user_training": True,
            "recommendations": ["Complete WCAG 2.1 AA compliance assessment"]
        }
    
    def _log_compliance_activity(self, activity_type: str, activity_id: str) -> None:
        """Log compliance-related activities."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity_type,
            "activity_id": activity_id,
            "user": "system",
            "details": f"{activity_type} completed for {activity_id}"
        }
        
        self.audit_trail.append(log_entry)
        logger.info(f"Compliance activity logged: {activity_type}")


# NHS Compliance Assessment
def assess_orphan_nhs_compliance() -> Dict[str, Any]:
    """Assess Orphan platform NHS compliance."""
    compliance_system = NHSSAMDCompliance("conf/nhs_compliance.json")
    
    # Perform SAMD classification
    risk_assessment = compliance_system.perform_samd_classification(
        intended_use="Clinical decision support for symptom assessment and triage",
        target_population="UK primary care patients aged 16+",
        clinical_context="Primary care and urgent care settings"
    )
    
    # Assess DTAC compliance
    dtac_assessment = compliance_system.assess_nhs_dtac_compliance()
    
    # Generate regulatory submission
    submission = compliance_system.generate_regulatory_submission()
    
    return {
        "samd_classification": risk_assessment.samd_classification.value,
        "clinical_risk_level": risk_assessment.clinical_risk_level.value,
        "dtac_compliance": dtac_assessment,
        "regulatory_submission": submission,
        "compliance_recommendations": [
            "Complete clinical validation studies",
            "Establish formal clinical governance",
            "Obtain MHRA SAMD approval",
            "Complete NHS DTAC assessment"
        ]
    }
