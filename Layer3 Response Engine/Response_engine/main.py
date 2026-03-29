"""
IMMUNEX — Layer 3: FastAPI Inference Service
==============================================
Main entry point for Layer 3. Receives a correlated alert from Layer 2,
runs the full pipeline, and returns an IncidentResponse.

Pipeline per request
--------------------
  POST /respond
    1. Validate AlertRequest (Pydantic)
    2. ResponseEngine.predict()     → ActionDecision
    3. SafetyVerifier.verify()      → VerificationResult
    4. LLM reasoning + HITL gate    → final_action_idx
    5. ActionExecutor.execute()     → ExecutionResult
    6. PlaybookGenerator.generate() → PlaybookReport
    7. AuditLogger.log_incident()   → compliance trail
    8. Return IncidentResponse

Endpoints
---------
  POST /respond
  POST /approve/{alert_id}
  GET  /pending
  GET  /health
  GET  /actions

Run
---
  uvicorn response_engine.main:app --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
import uuid
import heapq
from datetime import datetime, timezone
from typing import Any, Literal
from dataclasses import asdict

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# FIX 2: All imports at module level — not inside request handlers
from response_engine_module import ActionDecision, ResponseEngine
from safety_verifier import SafetyVerifier, VerificationResult
from action_executor import ActionExecutor, ExecutionResult
from playbook_generator import PlaybookGenerator, PlaybookReport
from audit_logger import AuditLogger
from action_registry import ACTION_NAMES, get_action_category
from llm_reasoning import generate_llm_reasoning, human_approval
from database import insert_incident_memory, init_pool, close_pool

# ── Env config ─────────────────────────────────────────────────────────────
_MODEL_PATH      = os.environ.get("IMMUNEX_MODEL_PATH",      "model_weights/dueling_dqn_immunex.zip")
_MGMT_IP         = os.environ.get("IMMUNEX_MGMT_IP",         "127.0.0.1")
_BACKUP_REGISTRY = os.environ.get("IMMUNEX_BACKUP_REGISTRY", "backups/registry.json")
_OLLAMA_HOST     = os.environ.get("IMMUNEX_OLLAMA_HOST",     "http://localhost:11434")
_DRY_RUN         = os.environ.get("IMMUNEX_DRY_RUN",         "true").lower() == "true"
_PORT            = int(os.environ.get("IMMUNEX_PORT",         "8001"))
_AUDIT_LOG_DIR   = os.environ.get("IMMUNEX_AUDIT_LOG_DIR",   "audit_logs")

# High-impact actions are now evaluated dynamically, all require approval

# ── Layer 4 retrain config ──────────────────────────────────────────────────
_L4_RETRAIN_URL    = os.environ.get("IMMUNEX_L4_URL", "http://localhost:8004") + "/retrain"
_L4_RETRAIN_ENABLE = os.environ.get("IMMUNEX_L4_RETRAIN", "true").lower() == "true"

# Attack types that are routine — skip retraining for these
_KNOWN_ATTACK_TYPES = {
    "benign", "dos", "ddos", "portscan", "bruteforce",
    "infiltration", "botnet", "web_attack", "unknown",
}


async def trigger_l4_retrain(alert: dict, feature_vector: list) -> None:
    """
    Fire-and-forget POST to L4 /retrain after L3 handles an incident.
    Only triggers for novel attack types or critical/high severity.
    Non-blocking — never raises into the main response path.

    Sends 25-dim slice of feature vector that L4 model expects.
    L4 mixes it with rehearsal data and retrains LoRA adapters in background.
    """
    if not _L4_RETRAIN_ENABLE:
        return

    attack_type = alert.get("attack_type", "unknown").lower().replace(" ", "_")
    is_novel    = attack_type not in _KNOWN_ATTACK_TYPES
    severity    = alert.get("severity", "low")

    # Only retrain for novel attacks OR critical/high severity
    if not is_novel and severity not in ("critical", "high"):
        return

    # L4 expects 25-dim — slice from the 128-dim L3 vector
    features_25 = feature_vector[:25] if len(feature_vector) >= 25 else (
        feature_vector + [0.0] * (25 - len(feature_vector))
    )

    payload = {
        "attack_features": [features_25],
        "attack_labels":   [1],
        "trigger_source":  f"layer3_auto_{attack_type}",
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(_L4_RETRAIN_URL, json=payload)
            if resp.status_code == 200:
                logger.info(
                    "l4_retrain_triggered",
                    attack_type = attack_type,
                    novel       = is_novel,
                    severity    = severity,
                    l4_response = resp.json().get("message"),
                )
            else:
                logger.warning(
                    "l4_retrain_rejected",
                    status_code = resp.status_code,
                    attack_type = attack_type,
                )
    except Exception as exc:
        # Never let L4 failure affect L3 response
        logger.warning("l4_retrain_failed", error=str(exc), attack_type=attack_type)


# ── Logging ────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger("immunex.api")

# ── Global pipeline components (initialised on startup) ───────────────────
_engine:    ResponseEngine    | None = None
_verifier:  SafetyVerifier    | None = None
_executor:  ActionExecutor    | None = None
_generator: PlaybookGenerator | None = None
_audit:     AuditLogger       | None = None  # FIX 3: AuditLogger now wired in

# PRIORITY SYSTEM: Dual store for fast lookups and sorted extractions
_priority_queue: list[tuple[int, str, str]] = []  # heapq of (priority, timestamp, alert_id)
_pending_lookup: dict[str, tuple[dict, Any, Any, dict, int, str]] = {} # alert_id -> (alert, decision, verification, reasoning, priority, timestamp)

class RateLimiter:
    """Mock Redis implementation."""
    def allow(self, key: str) -> bool:
        return True

_rate_limiter = RateLimiter()

def calculate_priority(alert: dict, decision: Any) -> int:
    """
    Calculates 1-4 bounds-checked priority based on severity mapping.
    1 = Critical, 4 = Low. Boosted by high impact/uncertainty.
    """
    severity = alert.get("severity", "medium").lower()
    base_priority = 4
    if severity == "critical":
        base_priority = 1
    elif severity == "high":
        base_priority = 2
    elif severity == "medium":
        base_priority = 3
    else:
        base_priority = 4
        
    if getattr(decision, "impact", "") == "high" or getattr(decision, "uncertain", False):
        base_priority = max(1, base_priority - 1)
        
    return base_priority


# ── Pydantic request / response models ────────────────────────────────────

class AlertRequest(BaseModel):
    model_config = {"extra": "ignore"}

    alert_id:           str
    timestamp:          str
    source_ip:          str
    destination_ip:     str
    source_port:        int
    destination_port:   int
    protocol:           str
    severity:           Literal["low", "medium", "high", "critical"]
    attack_type:        str
    feature_vector:     list[float]
    layer2_confidence:  float
    # Explicit 768D RoBERTa embedding carried from Layer 1 for pgvector storage
    roberta_embedding:  list[float] | None = None
    # MITRE context from Layer 2
    mitre_stage:        str | None = None
    predicted_next_stage: str | None = None

    @field_validator("feature_vector", mode="before")
    @classmethod
    def _normalize_feature_vector(cls, v):
        if not v:
            raise ValueError("feature_vector must not be empty")
        # RL model trained on 128-dim; truncate or pad to match
        TARGET = 128
        if len(v) > TARGET:
            return v[:TARGET]
        elif len(v) < TARGET:
            return list(v) + [0.0] * (TARGET - len(v))
        return v

    @field_validator("feature_vector")
    @classmethod
    def vector_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("feature_vector must not be empty")
        return v

    @field_validator("layer2_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("layer2_confidence must be in range [0.0, 1.0]")
        return v


class IncidentResponse(BaseModel):
    alert_id:             str
    status:               Literal["responded", "pending_approval", "failed", "rejected"]
    pipeline_duration_ms: float
    decision:             dict[str, Any]
    verification:         dict[str, Any]
    execution:            dict[str, Any] | None = None
    playbook:             dict[str, Any] | None = None
    llm_reasoning:        dict[str, Any] | None = None
    priority:             int | None = None
    uncertain:            bool | None = None
    impact:               str | None = None
    audit_entry_id:       str | None = None
    error:                str | None = None


class PendingItem(BaseModel):
    alert_id:         str
    actions:          list[str]
    severity:         str
    risk_level:       str
    priority:         int
    uncertain:        bool
    impact:           str
    timestamp_queued: str

class ApprovalRequest(BaseModel):
    approved:         bool
    modified_actions: list[int] | None = None


class ActionInfo(BaseModel):
    action_index:      int
    action_name:       str
    action_category:   str
    requires_approval: bool


class HealthResponse(BaseModel):
    status:           str
    model_loaded:     bool
    ollama_available: bool
    dry_run:          bool
    pending_count:    int


# ── Startup / shutdown lifecycle ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialises all pipeline components before the server accepts requests."""
    global _engine, _verifier, _executor, _generator, _audit

    logger.info("immunex_starting_up", model_path=_MODEL_PATH, dry_run=_DRY_RUN)

    # 1. Load DQN model
    try:
        _engine = ResponseEngine(
            model_path           = _MODEL_PATH,
        )
        logger.info("component_ready", component="ResponseEngine")
    except Exception as exc:
        logger.error("component_failed", component="ResponseEngine", error=str(exc))
        raise RuntimeError(f"Fatal: cannot load DQN model — {exc}") from exc

    # 1b. Database pool (pgvector incident memory)
    await init_pool()
    # 2. Safety verifier
    _verifier = SafetyVerifier(
        mgmt_ip              = _MGMT_IP,
        backup_registry_path = _BACKUP_REGISTRY,
    )
    logger.info("component_ready", component="SafetyVerifier")

    # 3. Action executor
    _executor = ActionExecutor(dry_run=_DRY_RUN)
    logger.info("component_ready", component="ActionExecutor", dry_run=_DRY_RUN)

    # 4. Playbook generator
    # FIX 3: Updated model name from "llama3" to "llama3:8b-instruct-q4_0"
    #        to match the quantised model pulled via `ollama pull`.
    _generator = PlaybookGenerator(model="llama3:8b-instruct-q4_0", ollama_host=_OLLAMA_HOST)
    logger.info(
        "component_ready",
        component        = "PlaybookGenerator",
        ollama_available = _generator._ollama_available,
    )

    # FIX 3: AuditLogger initialised at startup, not inside request handlers
    _audit = AuditLogger(log_dir=_AUDIT_LOG_DIR)
    logger.info("component_ready", component="AuditLogger", log_dir=_AUDIT_LOG_DIR)

    logger.info(
        "IMMUNEX Layer 3 ready",
        model_loaded     = True,
        ollama_available = _generator._ollama_available,
        dry_run          = _DRY_RUN,
        mgmt_ip          = _MGMT_IP,
        model_path       = _MODEL_PATH,
    )

    yield   # ← server is live here

    logger.info("immunex_shutting_down")
    await close_pool()


# ── FastAPI application ───────────────────────────────────────────────────

app = FastAPI(
    title       = "IMMUNEX Layer 3 — Immune Response Engine",
    description = (
        "Autonomous cyber incident response for banking environments. "
        "Receives correlated alerts from Layer 2 (LSTM), runs them through "
        "a Dueling DQN, Z3 safety verification, stub execution, and "
        "Llama-3 playbook generation."
    ),
    version  = "3.0.0",
    lifespan = lifespan,
    docs_url = "/docs",
    redoc_url= "/redoc",
)

# CORS — internal offline service, allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)


# ── 60-second request timeout middleware ──────────────────────────────────

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=60.0)
    except asyncio.TimeoutError:
        logger.warning(
            "request_timeout",
            path   = str(request.url.path),
            method = request.method,
        )
        return JSONResponse(
            status_code = 504,
            content     = {"error": "Request timed out after 60 seconds"},
        )


# ── X-Response-Time-Ms middleware ─────────────────────────────────────────

@app.middleware("http")
async def add_response_time_header(request: Request, call_next):
    t_start  = time.perf_counter()
    response: Response = await call_next(request)
    elapsed  = (time.perf_counter() - t_start) * 1000.0
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.3f}"
    return response


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@app.post(
    "/respond",
    response_model = IncidentResponse,
    tags           = ["Pipeline"],
    summary        = "Run the full Layer 3 incident response pipeline",
    status_code    = 200,
)
async def respond(alert_req: AlertRequest) -> IncidentResponse:
    """
    Accepts a correlated alert from Layer 2 and runs through:
    - DQN inference (ResponseEngine)
    - Z3 formal safety verification (SafetyVerifier)
    - LLM reasoning + human-in-the-loop gate
    - Action execution / simulation (ActionExecutor)
    - LLM playbook generation (PlaybookGenerator)
    - Immutable audit log entry (AuditLogger)
    """
    # FIX 1: t_start captured immediately — before any branch that uses elapsed_ms
    t_start  = time.perf_counter()
    alert_id = alert_req.alert_id
    alert    = alert_req.model_dump()
    
    # Check rate limit
    if not _rate_limiter.allow(alert.get("source_ip", "unknown")):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for source IP")

    if _engine is None or _verifier is None or _executor is None or _generator is None or _audit is None:
        return IncidentResponse(
            alert_id             = alert_id,
            status               = "failed",
            pipeline_duration_ms = 0.0,
            decision             = {},
            verification         = {},
            error                = "Pipeline components not initialised (startup failed?)",
        )

    try:
        # Step 2: DQN inference (returns Top-K)
        decision: ActionDecision = _engine.predict(alert)

        # Step 3: Z3 safety verification
        verification: VerificationResult = _verifier.verify(decision, alert)

        # Use substituted action if Z3 swapped it
        if verification.substituted_action is not None:
            decision = verification.substituted_action

        # --- LLM REASONING LAYER ---
        z3_status = "ALLOWED" if verification.approved else "BLOCKED"
        
        # Primary action name passed for reasoning, or a summary
        reasoning_target = ", ".join(decision.action_names)
        reasoning = await generate_llm_reasoning(alert, reasoning_target, verification.reason)
        risk      = reasoning.get("risk", "MEDIUM")

        # Flag for human attention if risk is HIGH
        if risk == "HIGH":
            decision.impact = "high"

        # Structured logging
        logger.info(
            "llm_advisory_generated",
            suggested_actions= decision.action_names,
            z3_status        = z3_status,
            llm_reason       = reasoning.get("reason"),
            risk_level       = risk,
        )
        
        # Priority Calculation
        decision.uncertain = decision.confidence < 0.6
        priority = calculate_priority(alert, decision)

        # ── AUTO-EXECUTE fast path ─────────────────────────────────────────
        # Low-risk, high-confidence actions skip the approval queue entirely.
        # This prevents the queue flooding with trivial monitoring actions.
        if risk == "LOW" and not decision.uncertain and z3_status == "ALLOWED":
            execution = _executor.execute(decision, alert, approval_granted=True)
            playbook  = _generator.generate(alert, decision, verification, execution)
            audit_entry_id = _audit.log_incident(
                alert=alert, decision=decision, verification=verification,
                execution=execution, playbook=playbook, priority=priority,
            )
            elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            logger.info(
                "low_risk_action_auto_executed",
                alert_id    = alert_id,
                actions     = decision.action_names,
                risk        = risk,
                priority    = priority,
                duration_ms = elapsed_ms,
            )
            # ── Trigger L4 adaptive retraining (fire-and-forget) ──────────
            asyncio.create_task(
                trigger_l4_retrain(alert, alert.get("feature_vector", []))
            )
            # ── Store in incident_memory (fire-and-forget) ──────────────────
            asyncio.create_task(insert_incident_memory(
                alert_id       = alert_id,
                source_ip      = alert.get("source_ip", ""),
                attack_type    = alert.get("attack_type", "unknown"),
                mitre_stage    = alert.get("mitre_stage", ""),
                predicted_next = alert.get("predicted_next_stage", ""),
                severity       = float(alert.get("layer2_confidence", 0.5)),
                god_mode_128d  = alert.get("feature_vector", []),
                roberta_768d   = alert.get("roberta_embedding") or [],
                playbook_text  = str(getattr(playbook, "steps", "")),
                action_taken   = decision.action_names,
                was_overridden = False,
                accepted_by    = "auto_system",
            ))
            return IncidentResponse(
                alert_id             = alert_id,
                status               = "responded",
                pipeline_duration_ms = elapsed_ms,
                decision             = asdict(decision),
                verification         = asdict(verification),
                execution            = asdict(execution),
                playbook             = asdict(playbook),
                llm_reasoning        = reasoning,
                priority             = priority,
                uncertain            = decision.uncertain,
                impact               = getattr(decision, "impact", "unknown"),
                audit_entry_id       = audit_entry_id,
            )

        # All MEDIUM/HIGH risk actions go to the pending approval queue.
        ts_queued = datetime.utcnow().isoformat() + "Z"
        _pending_lookup[alert_id] = (alert, decision, verification, reasoning, priority, ts_queued)
        heapq.heappush(_priority_queue, (priority, ts_queued, alert_id))

        # Log approval request to audit trail
        _audit.log_approval_request(
            alert_id     = alert_id,
            action_name  = ", ".join(decision.action_names),
            requested_at = ts_queued,
        )

        logger.info(
            "actions_queued_for_approval",
            alert_id    = alert_id,
            priority    = priority,
            actions     = decision.action_names,
            severity    = decision.severity,
            uncertain   = decision.uncertain,
        )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        return IncidentResponse(
            alert_id             = alert_id,
            status               = "pending_approval",
            pipeline_duration_ms = elapsed_ms,
            decision             = asdict(decision),
            verification         = asdict(verification),
            execution            = None,
            playbook             = None,
            llm_reasoning        = reasoning,
            priority             = priority,
            uncertain            = decision.uncertain,
            impact               = getattr(decision, "impact", "unknown"),
        )

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        logger.error("pipeline_exception", alert_id=alert_id, error=str(exc))
        return IncidentResponse(
            alert_id             = alert_id,
            status               = "failed",
            pipeline_duration_ms = elapsed_ms,
            decision             = {},
            verification         = {},
            error                = str(exc),
        )


@app.post(
    "/approve/{alert_id}",
    response_model = IncidentResponse,
    tags           = ["Approvals"],
    summary        = "Approve and execute a pending high-impact action",
)
async def approve(
    alert_id:    str,
    raw_request: Request,   # FIX: receive raw Request instead of a typed body param.
                            # This prevents FastAPI from running Pydantic validation
                            # (and returning 422) before we can do the 404 check.
) -> IncidentResponse:
    """
    Grants human approval for a queued action. Pops from the queue safely.
    Rejects or Modifies Actions depending on human operator input.
    Body is optional JSON matching ApprovalRequest; omitting it defaults to approved=True.
    """
    # ── 404 check FIRST — before any body work ─────────────────────────
    if alert_id not in _pending_lookup:
        raise HTTPException(
            status_code = 404,
            detail      = f"No pending action found for alert_id={alert_id}",
        )

    # ── Parse optional JSON body ───────────────────────────────────────
    req = ApprovalRequest(approved=True)        # safe default
    try:
        body_bytes = await raw_request.body()
        if body_bytes.strip():
            body_json = await raw_request.json()
            req = ApprovalRequest(**body_json)
    except Exception:
        pass    # empty / malformed body → keep default

    t_start = time.perf_counter()
    alert, decision, verification, reasoning, priority, ts = _pending_lookup.pop(alert_id)
    
    if not req.approved:
        # Request rejected. Do NOT pass to execution or playbook.
        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = False,
            approver = "human_operator",
            reason   = "Manually rejected via POST /approve endpoint",
        )
        # We don't execute, but we must log the incident termination
        audit_entry_id = _audit.log_incident(
            alert=alert, decision=decision, verification=verification,
            execution=None, playbook=None, modified_actions=req.modified_actions, priority=priority,
            before_actions=decision.actions.copy() if decision.actions else []
        )
        return IncidentResponse(
            alert_id             = alert_id,
            status               = "rejected",
            pipeline_duration_ms = round((time.perf_counter() - t_start) * 1000.0, 3),
            decision             = asdict(decision),
            verification         = asdict(verification),
            execution            = None,
            playbook             = None,
            llm_reasoning        = reasoning,
            priority             = priority,
            uncertain            = getattr(decision, "uncertain", False),
            impact               = getattr(decision, "impact", "unknown"),
        )
        
    before_actions = decision.actions.copy() if decision.actions else []
        
    # Apply human-modified actions over DQN
    if req.modified_actions:
        # Validate modified actions
        valid_ids = [idx for idx in req.modified_actions if idx in ACTION_NAMES]
        if valid_ids:
            decision.actions = valid_ids
            decision.action_names = [ACTION_NAMES.get(idx, "unknown") for idx in valid_ids]
            decision.action_categories = [get_action_category(idx) for idx in valid_ids]

    try:
        # 1. Playbook Override Workflow
        # Synthesize a pre-execution mock to safely generate the playbook prior to real execution
        from action_executor import ExecutionResult
        dummy_exec = ExecutionResult(
            alert_id=alert_id, action_index=decision.action_index, action_name=decision.action_name,
            actions=decision.actions, action_names=decision.action_names, status="simulated",
            dry_run=True, execution_time_ms=0.0, output={}, error=None, validation_status={}
        )
        playbook = _generator.generate(alert, decision, verification, dummy_exec)
        
        # Override execution sequence if Playbook specified valid exact executions
        if getattr(playbook, "executable_actions", None):
            if playbook.executable_actions:
                valid_playbook_ids = [idx for idx in playbook.executable_actions if idx in ACTION_NAMES]
                decision.actions = valid_playbook_ids
                decision.action_names = [ACTION_NAMES.get(idx, "unknown") for idx in decision.actions]
                decision.action_categories = [get_action_category(idx) for idx in decision.actions]
            
        # 2. Execution Run (Parameter validations will natively occur inside this loop)
        execution = _executor.execute(decision, alert, approval_granted=True)

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        # 3. Log approval trace properly
        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = True,
            approver = "human_operator",
            reason   = "Manually approved via POST /approve endpoint"
        )
        audit_entry_id = _audit.log_incident(
            alert        = alert,
            decision     = decision,
            verification = verification,
            execution    = execution,
            playbook     = playbook,
            modified_actions = req.modified_actions,
            priority     = priority,
            before_actions = before_actions
        )

        logger.info(
            "approved_action_executed",
            alert_id       = alert_id,
            actions        = decision.action_names,
            audit_entry_id = audit_entry_id,
            duration_ms    = elapsed_ms,
        )
        # ── Trigger L4 adaptive retraining (fire-and-forget) ──────────────
        asyncio.create_task(
            trigger_l4_retrain(alert, alert.get("feature_vector", []))
        )
        # ── Store in incident_memory (fire-and-forget) ──────────────────
        asyncio.create_task(insert_incident_memory(
            alert_id       = alert_id,
            source_ip      = alert.get("source_ip", ""),
            attack_type    = alert.get("attack_type", "unknown"),
            mitre_stage    = alert.get("mitre_stage", ""),
            predicted_next = alert.get("predicted_next_stage", ""),
            severity       = float(alert.get("layer2_confidence", 0.5)),
            god_mode_128d  = alert.get("feature_vector", []),
            roberta_768d   = alert.get("roberta_embedding") or [],
            playbook_text  = str(getattr(playbook, "steps", "")),
            action_taken   = decision.action_names,
            was_overridden = bool(req.modified_actions),
            override_actions = [ACTION_NAMES.get(i, str(i)) for i in (req.modified_actions or [])],
            accepted_by    = "human_operator",
        ))

        return IncidentResponse(
            alert_id             = alert_id,
            status               = "responded",
            pipeline_duration_ms = elapsed_ms,
            decision             = asdict(decision),
            verification         = asdict(verification),
            execution            = asdict(execution),
            playbook             = asdict(playbook),
            llm_reasoning        = reasoning,
            audit_entry_id       = audit_entry_id,
            priority             = priority,
            uncertain            = getattr(decision, "uncertain", False),
            impact               = getattr(decision, "impact", "unknown"),
        )

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        logger.error("approval_execution_failed", alert_id=alert_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(
    "/pending",
    response_model = list[PendingItem],
    tags           = ["Approvals"],
    summary        = "List all actions awaiting human approval",
)
async def get_pending() -> list[PendingItem]:
    """
    Returns all alerts currently held in the pending approval queue
    with action name, severity, and time queued.
    """
    global _priority_queue
    _priority_queue = [item for item in _priority_queue if item[2] in _pending_lookup]
    heapq.heapify(_priority_queue)
    
    result = []
    
    for prio, ts, aid in sorted(_priority_queue):
        item = _pending_lookup[aid]
        decision = item[1]
        reasoning = item[3]
        result.append(PendingItem(
                alert_id         = aid,
                actions          = decision.action_names,
                severity         = decision.severity,
                risk_level       = reasoning.get("risk", "UNKNOWN"),
                priority         = item[4],
                uncertain        = getattr(decision, "uncertain", False),
                impact           = getattr(decision, "impact", "UNKNOWN"),
                timestamp_queued = ts,
            ))
        
    return result


@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "Health check — confirms all components are ready",
)
async def health() -> HealthResponse:
    """
    Returns component status:
    - `model_loaded`: DQN model loaded successfully
    - `ollama_available`: Llama-3 server reachable
    - `dry_run`: whether the executor is in simulation mode
    - `pending_count`: actions awaiting human approval
    """
    return HealthResponse(
        status           = "ok",
        model_loaded     = _engine is not None,
        ollama_available = _generator is not None and _generator._ollama_available,
        dry_run          = _DRY_RUN,
        pending_count    = len(_pending_lookup),
    )


@app.get(
    "/actions",
    response_model = list[ActionInfo],
    tags           = ["System"],
    summary        = "List all 50 possible containment actions",
)
async def list_actions() -> list[ActionInfo]:
    """
    Returns the full registry of 50 discrete response actions with their
    index, name, category, and whether they require human approval.
    """
    return [
        ActionInfo(
            action_index      = idx,
            action_name       = name,
            action_category   = get_action_category(idx),
            requires_approval = True,
        )
        for idx, name in ACTION_NAMES.items()
    ]


# ── Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "response_engine.main:app",
        host      = "0.0.0.0",
        port      = _PORT,
        reload    = False,
        log_level = "info",
    )