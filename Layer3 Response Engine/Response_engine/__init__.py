"""
IMMUNEX — Layer 3: Immune Response Engine
==========================================
Autonomous cyber incident response pipeline.

Components
----------
ResponseEngine       — Dueling DQN inference (action selection)
SafetyVerifier       — Z3 formal constraint verification
ActionExecutor       — Stub executor for all 50 containment actions
PlaybookGenerator    — Offline Llama-3 SOC report generation
AuditLogger          — Immutable compliance audit trail
"""

__version__ = "3.0.0"
__author__  = "IMMUNEX — Team Wayfinders"

# Expose primary classes at package level so callers can write:
#   from response_engine import ResponseEngine, ActionDecision
# instead of:
#   from response_engine.response_engine import ResponseEngine

from response_engine.response_engine import ResponseEngine, ActionDecision
from response_engine.action_registry import ACTION_NAMES, get_action_category
from response_engine.safety_verifier import SafetyVerifier, VerificationResult

from response_engine.action_executor    import ActionExecutor, ExecutionResult
from response_engine.playbook_generator import PlaybookGenerator, PlaybookReport
from response_engine.audit_logger       import AuditLogger

__all__ = [
    "ResponseEngine",
    "ActionDecision",
    "SafetyVerifier",
    "VerificationResult",
    "ActionExecutor",
    "ExecutionResult",
    "PlaybookGenerator",
    "PlaybookReport",
    "AuditLogger",
    "ACTION_NAMES",
    "get_action_category",
]