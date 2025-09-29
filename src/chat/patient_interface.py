"""
Natural Language Patient Interface

Provides conversational interface for patients using WHAM and AS METHOD protocols
with natural language understanding and medical safety features.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from ..protocols.wham.protocol import WHAMProtocol, WHAMStage
from ..protocols.as_method.protocol import ASMethodProtocol, ASMethodStage
from ..protocols.safety.red_flags import RedFlagDetector
from ..ontology.snomed.loader import SNOMEDLoader
from ..utils.logging.logger import get_logger

logger = get_logger(__name__)


class ConversationMode(Enum):
    """Conversation modes for patient interface."""
    WHAM = "wham"
    AS_METHOD = "as_method"
    FREE_FORM = "free_form"


@dataclass
class ConversationState:
    """Current state of patient conversation."""
    mode: ConversationMode
    stage: Optional[str] = None
    collected_info: Dict[str, Any] = None
    red_flags_detected: List[str] = None
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.collected_info is None:
            self.collected_info = {}
        if self.red_flags_detected is None:
            self.red_flags_detected = []
        if self.conversation_history is None:
            self.conversation_history = []


class PatientInterface:
    """
    Natural language patient interface with medical protocol integration.
    
    Features:
    - WHAM and AS METHOD protocol support
    - Natural language understanding
    - Red flag detection and safety alerts
    - Conversational flow management
    - Medical terminology recognition
    - Patient-friendly communication
    """
    
    def __init__(self, default_mode: ConversationMode = ConversationMode.WHAM):
        self.wham_protocol = WHAMProtocol()
        self.as_method_protocol = ASMethodProtocol()
        self.red_flag_detector = RedFlagDetector()
        self.snomed_loader = SNOMEDLoader()
        
        self.conversation_state = ConversationState(mode=default_mode)
        self.greeting_given = False
        
        # Natural language patterns for intent recognition
        self.intent_patterns = self._initialize_intent_patterns()
        
        logger.info(f"Patient interface initialized with {default_mode.value} mode")
    
    def _initialize_intent_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for recognizing patient intents."""
        return {
            "greeting": [
                r'\b(?:hello|hi|hey|good\s+(?:morning|afternoon|evening))\b',
                r'\b(?:i\s+need\s+help|can\s+you\s+help)\b'
            ],
            "symptom_description": [
                r'\b(?:i\s+have|i\'m\s+experiencing|i\s+feel|my\s+\w+\s+(?:hurts?|aches?))\b',
                r'\b(?:pain|ache|sore|hurt|uncomfortable|sick|unwell)\b'
            ],
            "question": [
                r'\b(?:what|how|when|where|why|should\s+i|can\s+i|is\s+it)\b',
                r'\?'
            ],
            "emergency": [
                r'\b(?:emergency|urgent|help|911|999|ambulance)\b',
                r'\b(?:can\'?t\s+breathe|chest\s+pain|severe\s+pain)\b'
            ],
            "medication_query": [
                r'\b(?:medication|medicine|drug|pill|tablet|taking)\b',
                r'\b(?:side\s+effect|interaction|dosage|dose)\b'
            ]
        }
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process patient message and generate appropriate response.
        
        Args:
            user_message: Patient's input message
            
        Returns:
            Response dictionary with message, questions, and metadata
        """
        # Add to conversation history
        self.conversation_state.conversation_history.append({
            "role": "user",
            "message": user_message,
            "timestamp": self._get_timestamp()
        })
        
        # Check for red flags first
        red_flags = self.red_flag_detector.detect_red_flags(user_message)
        if red_flags:
            return self._handle_red_flags(red_flags, user_message)
        
        # Handle greeting if not given
        if not self.greeting_given:
            return self._handle_initial_greeting(user_message)
        
        # Recognize intent
        intent = self._recognize_intent(user_message)
        
        # Process based on conversation mode and intent
        if self.conversation_state.mode == ConversationMode.WHAM:
            return self._process_wham_conversation(user_message, intent)
        elif self.conversation_state.mode == ConversationMode.AS_METHOD:
            return self._process_as_method_conversation(user_message, intent)
        else:
            return self._process_free_form_conversation(user_message, intent)
    
    def _handle_initial_greeting(self, user_message: str) -> Dict[str, Any]:
        """Handle initial patient greeting and setup."""
        self.greeting_given = True
        
        greeting_response = (
            "Hello! I'm here to help you with your health concerns. "
            "I'll ask you some questions to better understand your situation and provide appropriate guidance.\n\n"
            "Please remember that I'm here to provide information and support, but I cannot replace "
            "professional medical advice. If you're experiencing a medical emergency, please call "
            "emergency services immediately (999 in the UK, 911 in the US).\n\n"
            "Let's start - what brings you here today?"
        )
        
        response = {
            "message": greeting_response,
            "questions": ["What symptoms or health concerns would you like to discuss?"],
            "conversation_stage": "initial_greeting",
            "safety_message": "For emergencies, call 999 (UK) or 911 (US) immediately",
            "next_action": "await_symptom_description"
        }
        
        self._add_to_conversation_history("assistant", greeting_response)
        return response
    
    def _handle_red_flags(self, red_flags: List, user_message: str) -> Dict[str, Any]:
        """Handle detected red flags with appropriate urgency."""
        emergency_response = self.red_flag_detector.get_emergency_response(red_flags)
        
        if emergency_response["emergency_level"] == "critical":
            response_message = (
                "🚨 **MEDICAL EMERGENCY DETECTED** 🚨\n\n"
                f"{emergency_response['message']}\n\n"
                "Based on what you've described, this requires immediate emergency medical attention. "
                "Please:\n"
                "• Call emergency services NOW (999 in UK, 911 in US)\n"
                "• Do not drive yourself to hospital\n"
                "• Stay on the line with emergency services\n"
                "• If possible, have someone stay with you\n\n"
                "This is not a situation for self-treatment or delay."
            )
        elif emergency_response["emergency_level"] == "high":
            response_message = (
                "⚠️ **URGENT MEDICAL ATTENTION NEEDED** ⚠️\n\n"
                f"{emergency_response['message']}\n\n"
                "Based on your symptoms, you should seek urgent medical care today. "
                "Please contact:\n"
                "• Your GP for an urgent appointment\n"
                "• NHS 111 (UK) for guidance\n"
                "• Walk-in centre or urgent care\n"
                "• A&E if other services unavailable\n\n"
                "Do not wait to see if symptoms improve."
            )
        else:
            response_message = (
                "⚠️ **MEDICAL ATTENTION RECOMMENDED** ⚠️\n\n"
                f"{emergency_response['message']}\n\n"
                "While not immediately life-threatening, your symptoms warrant medical review. "
                "Please arrange to see a healthcare professional today if possible."
            )
        
        response = {
            "message": response_message,
            "red_flags": [f.description for f in red_flags],
            "emergency_level": emergency_response["emergency_level"],
            "recommendations": emergency_response.get("recommendations", []),
            "conversation_stage": "emergency_detected",
            "requires_immediate_action": emergency_response["emergency_level"] == "critical"
        }
        
        self._add_to_conversation_history("assistant", response_message)
        return response
    
    def _recognize_intent(self, message: str) -> str:
        """Recognize patient intent from message."""
        message_lower = message.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent
        
        return "general"
    
    def _process_wham_conversation(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Process conversation using WHAM protocol."""
        # Process with WHAM protocol
        wham_response = self.wham_protocol.process_patient_input(user_message)
        
        # Generate natural language response
        stage = wham_response["stage"]
        stage_complete = wham_response.get("stage_complete", False)
        
        if stage == "WHO":
            response_message = self._generate_who_response(wham_response)
        elif stage == "HOW":
            response_message = self._generate_how_response(wham_response)
        elif stage == "ACTION":
            response_message = self._generate_action_response(wham_response)
        elif stage == "MONITORING":
            response_message = self._generate_monitoring_response(wham_response)
        else:
            response_message = "Thank you for that information. Let me ask you a few more questions."
        
        response = {
            "message": response_message,
            "questions": wham_response.get("next_questions", []),
            "conversation_stage": f"wham_{stage.lower()}",
            "stage_complete": stage_complete,
            "extracted_info": wham_response.get("extracted_info", {}),
            "protocol": "WHAM"
        }
        
        # Add red flag warnings if detected
        if "red_flags" in wham_response and wham_response["red_flags"]:
            response["red_flags_detected"] = wham_response["red_flags"]
        
        self._add_to_conversation_history("assistant", response_message)
        return response
    
    def _process_as_method_conversation(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Process conversation using AS METHOD protocol."""
        as_response = self.as_method_protocol.process_patient_input(user_message)
        
        stage = as_response["stage"]
        response_message = f"Thank you. {as_response.get('next_questions', [''])[0]}"
        
        response = {
            "message": response_message,
            "questions": as_response.get("next_questions", []),
            "conversation_stage": f"as_method_{stage.lower()}",
            "stage_complete": as_response.get("stage_complete", False),
            "extracted_info": as_response.get("extracted_info", {}),
            "protocol": "AS_METHOD"
        }
        
        self._add_to_conversation_history("assistant", response_message)
        return response
    
    def _process_free_form_conversation(self, user_message: str, intent: str) -> Dict[str, Any]:
        """Process free-form conversation."""
        if intent == "symptom_description":
            response_message = (
                "I understand you're experiencing some symptoms. To help you better, "
                "I'd like to ask some structured questions. This will help ensure I "
                "don't miss anything important."
            )
            
            # Switch to WHAM mode
            self.conversation_state.mode = ConversationMode.WHAM
            return self._process_wham_conversation(user_message, intent)
        
        elif intent == "question":
            response_message = (
                "I'd be happy to help answer your question. However, to provide the most "
                "accurate and safe guidance, I'd first like to understand your specific "
                "situation better."
            )
        
        else:
            response_message = (
                "Thank you for sharing that with me. To provide you with the best possible "
                "guidance, I'd like to ask you some questions about your health concern."
            )
        
        response = {
            "message": response_message,
            "questions": ["What specific symptoms or health concerns would you like to discuss?"],
            "conversation_stage": "free_form",
            "protocol": "FREE_FORM"
        }
        
        self._add_to_conversation_history("assistant", response_message)
        return response
    
    def _generate_who_response(self, wham_response: Dict) -> str:
        """Generate natural language response for WHO stage."""
        extracted = wham_response.get("extracted_info", {})
        
        response_parts = []
        
        if extracted.get("age"):
            response_parts.append(f"I see you're {extracted['age']} years old.")
        
        if extracted.get("gender"):
            response_parts.append("Thank you for that information.")
        
        if extracted.get("medications"):
            meds = extracted["medications"]
            if len(meds) == 1:
                response_parts.append(f"I note you're taking {meds[0]}.")
            elif len(meds) > 1:
                response_parts.append(f"I note you're taking {', '.join(meds[:-1])} and {meds[-1]}.")
        
        if wham_response.get("stage_complete"):
            response_parts.append("Now let's talk about your symptoms.")
        
        return " ".join(response_parts) if response_parts else "Thank you for that information."
    
    def _generate_how_response(self, wham_response: Dict) -> str:
        """Generate natural language response for HOW stage."""
        extracted = wham_response.get("extracted_info", {})
        
        response_parts = []
        
        if extracted.get("primary_complaint"):
            response_parts.append("I understand your main concern.")
        
        if extracted.get("duration"):
            response_parts.append(f"So this has been going on for {extracted['duration']}.")
        
        if extracted.get("severity"):
            severity = extracted["severity"]
            if severity >= 8:
                response_parts.append("That sounds quite severe.")
            elif severity >= 5:
                response_parts.append("That sounds moderately uncomfortable.")
            else:
                response_parts.append("I understand it's bothering you.")
        
        if wham_response.get("red_flags"):
            response_parts.append("I'm noting some concerning symptoms that may need urgent attention.")
        
        return " ".join(response_parts) if response_parts else "Thank you for describing your symptoms."
    
    def _generate_action_response(self, wham_response: Dict) -> str:
        """Generate natural language response for ACTION stage."""
        recommendations = wham_response.get("recommendations", [])
        
        if not recommendations:
            return "Let me provide you with some recommendations based on what you've told me."
        
        response_parts = ["Based on your symptoms, here's what I recommend:"]
        
        for i, rec in enumerate(recommendations, 1):
            response_parts.append(f"\n{i}. {rec['recommendation']}")
            if rec.get('rationale'):
                response_parts.append(f"   Reason: {rec['rationale']}")
        
        urgency = wham_response.get("referral_urgency", "routine")
        if urgency == "immediate":
            response_parts.append("\n⚠️ This requires immediate medical attention.")
        elif urgency == "urgent":
            response_parts.append("\n⚠️ Please seek medical attention today.")
        
        return "".join(response_parts)
    
    def _generate_monitoring_response(self, wham_response: Dict) -> str:
        """Generate natural language response for MONITORING stage."""
        safety_netting = wham_response.get("safety_netting", [])
        
        response_parts = ["Important safety information:"]
        
        for advice in safety_netting:
            response_parts.append(f"\n• {advice}")
        
        response_parts.append("\n\nDo you have any questions about this guidance?")
        
        return "".join(response_parts)
    
    def _add_to_conversation_history(self, role: str, message: str):
        """Add message to conversation history."""
        self.conversation_state.conversation_history.append({
            "role": role,
            "message": message,
            "timestamp": self._get_timestamp()
        })
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation."""
        if self.conversation_state.mode == ConversationMode.WHAM:
            assessment = self.wham_protocol.get_complete_assessment()
        elif self.conversation_state.mode == ConversationMode.AS_METHOD:
            assessment = self.as_method_protocol.get_complete_assessment()
        else:
            assessment = {}
        
        return {
            "conversation_mode": self.conversation_state.mode.value,
            "assessment": assessment,
            "red_flags_detected": self.conversation_state.red_flags_detected,
            "conversation_history": self.conversation_state.conversation_history,
            "total_messages": len(self.conversation_state.conversation_history)
        }
    
    def reset_conversation(self, mode: Optional[ConversationMode] = None):
        """Reset conversation state."""
        if mode:
            self.conversation_state.mode = mode
        
        self.conversation_state.stage = None
        self.conversation_state.collected_info = {}
        self.conversation_state.red_flags_detected = []
        self.conversation_state.conversation_history = []
        
        self.wham_protocol = WHAMProtocol()
        self.as_method_protocol = ASMethodProtocol()
        self.greeting_given = False
        
        logger.info("Conversation reset")
