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
from dataclasses import asdict
from datetime import datetime
from typing import Any, Literal

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# FIX 2: All imports at module level — not inside request handlers
from response_engine.response_engine import ActionDecision, ResponseEngine
from response_engine.safety_verifier import SafetyVerifier, VerificationResult
from response_engine.action_executor import ActionExecutor, ExecutionResult
from response_engine.playbook_generator import PlaybookGenerator, PlaybookReport
from response_engine.audit_logger import AuditLogger
from response_engine.action_registry import ACTION_NAMES, get_action_category
from response_engine.llm_reasoning import generate_llm_reasoning, human_approval

# ── Env config ─────────────────────────────────────────────────────────────
_MODEL_PATH      = os.environ.get("IMMUNEX_MODEL_PATH",      "models/dueling_dqn_immunex.zip")
_MGMT_IP         = os.environ.get("IMMUNEX_MGMT_IP",         "127.0.0.1")
_BACKUP_REGISTRY = os.environ.get("IMMUNEX_BACKUP_REGISTRY", "backups/registry.json")
_OLLAMA_HOST     = os.environ.get("IMMUNEX_OLLAMA_HOST",     "http://localhost:11434")
_DRY_RUN         = os.environ.get("IMMUNEX_DRY_RUN",         "true").lower() == "true"
_PORT            = int(os.environ.get("IMMUNEX_PORT",         "8001"))
_AUDIT_LOG_DIR   = os.environ.get("IMMUNEX_AUDIT_LOG_DIR",   "audit_logs")

# High-impact actions that require human approval
_HIGH_IMPACT_ACTIONS: list[int] = [12, 13, 14, 18, 22, 28, 30, 32, 33, 34, 46]

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

# In-memory pending approval queue:
#   alert_id → (alert_dict, ActionDecision, VerificationResult, timestamp)
_pending: dict[str, tuple[dict, ActionDecision, VerificationResult, str]] = {}


# ── Pydantic request / response models ────────────────────────────────────

class AlertRequest(BaseModel):
    model_config = {"extra": "forbid"}

    alert_id:          str
    timestamp:         str
    source_ip:         str
    destination_ip:    str
    source_port:       int
    destination_port:  int
    protocol:          str
    severity:          Literal["low", "medium", "high", "critical"]
    attack_type:       str
    feature_vector:    list[float]
    layer2_confidence: float

    @field_validator("feature_vector")
    @classmethod
    def vector_must_be_128(cls, v: list[float]) -> list[float]:
        if len(v) != 128:
            raise ValueError(
                f"feature_vector must have exactly 128 elements, got {len(v)}"
            )
        return v

    @field_validator("layer2_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("layer2_confidence must be in range [0.0, 1.0]")
        return v


class IncidentResponse(BaseModel):
    alert_id:             str
    status:               Literal["responded", "pending_approval", "failed"]
    pipeline_duration_ms: float
    decision:             dict[str, Any]
    verification:         dict[str, Any]
    execution:            dict[str, Any] | None = None
    playbook:             dict[str, Any] | None = None
    audit_entry_id:       str | None = None
    error:                str | None = None


class PendingItem(BaseModel):
    alert_id:         str
    action_name:      str
    severity:         str
    timestamp_queued: str


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
            require_approval_for = _HIGH_IMPACT_ACTIONS,
        )
        logger.info("component_ready", component="ResponseEngine")
    except Exception as exc:
        logger.error("component_failed", component="ResponseEngine", error=str(exc))
        raise RuntimeError(f"Fatal: cannot load DQN model — {exc}") from exc

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
        # Step 2: DQN inference
        decision: ActionDecision = _engine.predict(alert)

        # Step 3: Z3 safety verification
        verification: VerificationResult = _verifier.verify(decision, alert)

        # Use substituted action if Z3 swapped it (e.g. C5 rollback → backup)
        if verification.substituted_action is not None:
            decision = verification.substituted_action

        # --- HITL + LLM REASONING LAYER ---
        z3_status      = "ALLOWED" if verification.approved else "BLOCKED"
        human_decision = "N/A"
        reasoning      = {}
        rejected_once  = False

        if z3_status == "BLOCKED":
            final_action_idx = 0
            human_decision   = "Z3_BLOCKED"
        else:
            reasoning = await generate_llm_reasoning(alert, decision.action_name, verification.reason)
            risk      = reasoning.get("risk", "MEDIUM")

            if risk == "LOW":
                approved       = True
                human_decision = "AUTO_APPROVED"
            else:
                approved       = await human_approval(decision.action_name, reasoning)
                if not approved:
                    rejected_once = True
                human_decision = "APPROVED" if approved else "REJECTED"

            if approved:
                final_action_idx = decision.action_index
            else:
                alt       = reasoning.get("alternative", "do_nothing")
                found_idx = 0
                for idx, name in ACTION_NAMES.items():
                    if name == alt:
                        found_idx = idx
                        break
                final_action_idx = found_idx

        # Structured logging
        logger.info(
            "hitl_llm_decision",
            suggested_action = decision.action_name,
            z3_status        = z3_status,
            llm_reason       = reasoning.get("reason"),
            risk_level       = reasoning.get("risk"),
            human_decision   = human_decision,
            final_action     = ACTION_NAMES.get(final_action_idx, "do_nothing"),
        )

        # If rejected/blocked, override decision fields and bypass approval queue
        if final_action_idx != decision.action_index or rejected_once or z3_status == "BLOCKED":
            decision.action_index      = final_action_idx
            decision.action_name       = ACTION_NAMES.get(final_action_idx, "do_nothing")
            decision.action_category   = get_action_category(final_action_idx)
            decision.requires_approval = False
            verification.approved                = True
            verification.requires_human_approval = False

        # Step 4: Approval gate — queue and return 200 with pending_approval status
        if verification.requires_human_approval and not verification.approved:
            ts_queued = datetime.utcnow().isoformat() + "Z"
            _pending[alert_id] = (alert, decision, verification, ts_queued)

            # FIX 3: Log approval request to audit trail
            _audit.log_approval_request(
                alert_id     = alert_id,
                action_name  = decision.action_name,
                requested_at = ts_queued,
            )

            logger.info(
                "action_queued_for_approval",
                alert_id    = alert_id,
                action_name = decision.action_name,
                severity    = decision.severity,
            )

            # FIX 1: elapsed_ms is always defined here because t_start was set at top
            elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
            return IncidentResponse(
                alert_id             = alert_id,
                status               = "pending_approval",
                pipeline_duration_ms = elapsed_ms,
                decision             = asdict(decision),
                verification         = asdict(verification),
                execution            = None,
                playbook             = None,
            )

        # Step 5: Execute
        execution: ExecutionResult = _executor.execute(
            decision, alert, approval_granted=verification.approved
        )

        # Step 6: Playbook generation
        playbook: PlaybookReport = _generator.generate(
            alert, decision, verification, execution
        )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        # FIX 3: Write immutable audit entry for every completed incident
        audit_entry_id = _audit.log_incident(
            alert        = alert,
            decision     = decision,
            verification = verification,
            execution    = execution,
            playbook     = playbook,
        )

        logger.info(
            "pipeline_complete",
            alert_id             = alert_id,
            action_name          = decision.action_name,
            execution_status     = execution.status,
            audit_entry_id       = audit_entry_id,
            pipeline_duration_ms = elapsed_ms,
        )

        return IncidentResponse(
            alert_id             = alert_id,
            status               = "responded",
            pipeline_duration_ms = elapsed_ms,
            decision             = asdict(decision),
            verification         = asdict(verification),
            execution            = asdict(execution),
            playbook             = asdict(playbook),
            audit_entry_id       = audit_entry_id,
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
async def approve(alert_id: str) -> IncidentResponse:
    """
    Grants human approval for a queued high-impact action.
    Retrieves the action from the pending queue, executes it,
    generates the incident playbook, and writes an audit entry.

    Returns **404** if `alert_id` is not in the pending queue.
    """
    if alert_id not in _pending:
        raise HTTPException(
            status_code = 404,
            detail      = f"No pending action found for alert_id={alert_id}",
        )

    t_start = time.perf_counter()
    alert, decision, verification, _ = _pending.pop(alert_id)

    try:
        execution: ExecutionResult = _executor.execute(
            decision, alert, approval_granted=True
        )
        playbook: PlaybookReport = _generator.generate(
            alert, decision, verification, execution
        )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        # FIX 3: Log approval decision + full incident to audit trail
        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = True,
            approver = "human_operator",
            reason   = "Manually approved via POST /approve endpoint",
        )
        audit_entry_id = _audit.log_incident(
            alert        = alert,
            decision     = decision,
            verification = verification,
            execution    = execution,
            playbook     = playbook,
        )

        logger.info(
            "approved_action_executed",
            alert_id       = alert_id,
            action_name    = decision.action_name,
            audit_entry_id = audit_entry_id,
            duration_ms    = elapsed_ms,
        )

        return IncidentResponse(
            alert_id             = alert_id,
            status               = "responded",
            pipeline_duration_ms = elapsed_ms,
            decision             = asdict(decision),
            verification         = asdict(verification),
            execution            = asdict(execution),
            playbook             = asdict(playbook),
            audit_entry_id       = audit_entry_id,
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
    return [
        PendingItem(
            alert_id         = aid,
            action_name      = decision.action_name,
            severity         = decision.severity,
            timestamp_queued = ts,
        )
        for aid, (_, decision, __, ts) in _pending.items()
    ]


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
        pending_count    = len(_pending),
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
            requires_approval = idx in _HIGH_IMPACT_ACTIONS,
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