"""
IMMUNEX — Layer 3: Immune Response Engine
==========================================
Autonomous cyber incident response pipeline for banking environments.

Components
----------
ResponseEngine       — Dueling DQN inference (action selection)
SafetyVerifier       — Z3 formal constraint verification (6 constraints)
ActionExecutor       — Dispatcher for all 50 containment action stubs
PlaybookGenerator    — Offline Llama-3 SOC report generation + fallback
AuditLogger          — Immutable JSON Lines compliance audit trail
LLM Reasoning        — Risk classification + human-in-the-loop gate

Usage
-----
  from response_engine_module import ResponseEngine, ActionDecision
  from response_engine_module import SafetyVerifier, VerificationResult
  from response_engine_module import ActionExecutor, ExecutionResult
  from response_engine_module import PlaybookGenerator, PlaybookReport
  from response_engine_module import AuditLogger
  from response_engine_module import ACTION_NAMES, get_action_category
  from response_engine import ResponseEngine, ActionDecision
  from response_engine import SafetyVerifier, VerificationResult
  from response_engine import ActionExecutor, ExecutionResult
  from response_engine import PlaybookGenerator, PlaybookReport
  from response_engine import AuditLogger
  from response_engine import ACTION_NAMES, get_action_category
"""

__version__ = "3.0.0"
__author__  = "IMMUNEX — Team Wayfinders"

from response_engine.response_engine     import ResponseEngine, ActionDecision
from response_engine.action_registry     import ACTION_NAMES, get_action_category
from response_engine.safety_verifier     import SafetyVerifier, VerificationResult
from response_engine.action_executor     import ActionExecutor, ExecutionResult
from response_engine.playbook_generator  import PlaybookGenerator, PlaybookReport
from response_engine.audit_logger        import AuditLogger

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