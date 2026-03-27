"""
IMMUNEX — Layer 3: FastAPI Inference Service
==============================================
Main entry point for Layer 3. Receives a correlated alert from Layer 2,
runs the full pipeline, and returns an IncidentResponse.

Pipeline per request
--------------------
  POST /respond
    1. Validate AlertRequest (Pydantic)
    2. ResponseEngine.predict()     → ActionDecision   [async]
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
  POST /admin/reject
  POST /admin/override
  GET  /pending
  GET  /health
  GET  /actions

Run
---
  uvicorn response_engine.main:app --port 8001 --reload

[IMMUNEX-PATCH] Step 4 changes:
  Bug 7  — removed LOW-risk auto-execute; ALL alerts go to pending queue
  Bug 8  — /approve enforces playbook.executable_actions ⊆ dqn_actions
  Bug 9  — _pending_lookup.pop(alert_id, None) everywhere (race-condition safe)
  Bug 10 — background TTL sweeper (60 s interval, 1 h TTL)
  New    — POST /admin/reject (RLHF negative memory)
  New    — POST /admin/override (Z3 Pass 2 with is_human_reviewed=True)
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

# ── Env config ─────────────────────────────────────────────────────────────
_MODEL_PATH      = os.environ.get("IMMUNEX_MODEL_PATH",      "model_weights/dueling_dqn_immunex.zip")
_MGMT_IP         = os.environ.get("IMMUNEX_MGMT_IP",         "127.0.0.1")
_BACKUP_REGISTRY = os.environ.get("IMMUNEX_BACKUP_REGISTRY", "backups/registry.json")
_OLLAMA_HOST     = os.environ.get("IMMUNEX_OLLAMA_HOST",     "http://localhost:11434")
_DRY_RUN         = os.environ.get("IMMUNEX_DRY_RUN",         "true").lower() == "true"
_PORT            = int(os.environ.get("IMMUNEX_PORT",         "8001"))
_AUDIT_LOG_DIR   = os.environ.get("IMMUNEX_AUDIT_LOG_DIR",   "audit_logs")

# [IMMUNEX-PATCH] Bug 10: TTL sweeper parameters
_TTL_SWEEP_INTERVAL_S: int = 60       # sweep every 60 seconds
_TTL_MAX_AGE_S:        int = 3_600    # kill entries older than 1 hour

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


class ApprovalRequest(BaseModel):
    approved:         bool
    modified_actions: list[int] | None = None


# [IMMUNEX-PATCH] New model for POST /admin/reject
class RejectRequest(BaseModel):
    alert_id:         str
    rejection_reason: str  = "Rejected by human operator"
    admin_id:         str  = "human_operator"


# [IMMUNEX-PATCH] New model for POST /admin/override
class OverrideRequest(BaseModel):
    alert_id:        str
    override_actions: list[int]
    admin_id:        str = "human_operator"


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
    global _engine, _verifier, _executor, _generator, _audit, _ttl_task

    logger.info("immunex_starting_up", model_path=_MODEL_PATH, dry_run=_DRY_RUN)

    # [IMMUNEX-PATCH] Step 1: initialise asyncpg pool + pgvector schema
    await _db.init_pool()
    logger.info("component_ready", component="DatabasePool")

    # 1. Load DQN model
    try:
        _engine = ResponseEngine(model_path=_MODEL_PATH)
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

    # 5. Audit logger
    _audit = AuditLogger(log_dir=_AUDIT_LOG_DIR)
    logger.info("component_ready", component="AuditLogger", log_dir=_AUDIT_LOG_DIR)

    # [IMMUNEX-PATCH] Bug 10: start background TTL sweeper task
    _ttl_task = asyncio.create_task(_ttl_sweeper())
    logger.info("component_ready", component="TTLSweeper",
                interval_s=_TTL_SWEEP_INTERVAL_S, max_age_s=_TTL_MAX_AGE_S)

    logger.info(
        "IMMUNEX Layer 3 ready",
        model_loaded     = True,
        ollama_available = _generator._ollama_available,
        dry_run          = _DRY_RUN,
        mgmt_ip          = _MGMT_IP,
        model_path       = _MODEL_PATH,
    )

    yield   # ← server is live here

    # [IMMUNEX-PATCH] Bug 10: cancel TTL sweeper on shutdown
    if _ttl_task and not _ttl_task.done():
        _ttl_task.cancel()
        try:
            await _ttl_task
        except asyncio.CancelledError:
            pass

    # [IMMUNEX-PATCH] Step 1: close asyncpg pool gracefully
    await _db.close_pool()
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
    version  = "3.1.0",
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

    # Check API-level rate limit
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
        # Step 2: DQN inference (async — includes rejection-memory lookup)
        # [IMMUNEX-PATCH] predict() is now async due to asyncpg rejection check
        decision: ActionDecision = await _engine.predict(alert)

        # Step 3: Z3 safety verification
        # [IMMUNEX-PATCH] Bug 4+6: pass is_human_reviewed=False and
        # request_arrival_time from endpoint entry (before any queue delay)
        verification: VerificationResult = _verifier.verify(
            decision,
            alert,
            is_human_reviewed    = False,          # [IMMUNEX-PATCH] Bug 4: not yet reviewed
            request_arrival_time = request_arrival_time,  # [IMMUNEX-PATCH] Bug 6
        )

        # Use substituted action if Z3 swapped it
        if verification.substituted_action is not None:
            decision = verification.substituted_action

        # --- LLM REASONING LAYER ---
        z3_status = "ALLOWED" if verification.approved else "BLOCKED"
        reasoning_target = ", ".join(decision.action_names)
        # [IMMUNEX-PATCH] Bug 12: auto_approved=False (goes to queue), human_reviewed=False
        reasoning = await generate_llm_reasoning(
            alert,
            reasoning_target,
            verification.reason,
            auto_approved  = False,    # [IMMUNEX-PATCH] Bug 12
            human_reviewed = False,    # [IMMUNEX-PATCH] Bug 12
        )
        risk = reasoning.get("risk", "MEDIUM")

        if risk == "HIGH":
            decision.impact = "high"

        logger.info(
            "llm_advisory_generated",
            suggested_actions = decision.action_names,
            z3_status         = z3_status,
            llm_reason        = reasoning.get("reason"),
            risk_level        = risk,
        )

        decision.uncertain = decision.confidence < 0.6
        priority = calculate_priority(alert, decision)

        # [IMMUNEX-PATCH] Bug 7: NO auto-execute path for LOW risk.
        # ALL alerts — regardless of risk level — enter the pending queue.
        # This enforces the "ALL actions require approval" constraint.
        ts_iso       = datetime.utcnow().isoformat() + "Z"
        arrival_mono = time.monotonic()  # [IMMUNEX-PATCH] Bug 10: for TTL sweeper

        # Store arrival_mono as 7th element for TTL sweeper
        _pending_lookup[alert_id] = (
            alert, decision, verification, reasoning, priority, ts_iso, arrival_mono
        )
        heapq.heappush(_priority_queue, (priority, ts_iso, alert_id))

        # Log approval request to audit trail
        _audit.log_approval_request(
            alert_id     = alert_id,
            action_name  = ", ".join(decision.action_names),
            requested_at = ts_iso,
        )

        logger.info(
            "actions_queued_for_approval",
            alert_id  = alert_id,
            priority  = priority,
            actions   = decision.action_names,
            severity  = decision.severity,
            uncertain = decision.uncertain,
            risk      = risk,
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
    raw_request: Request,
) -> IncidentResponse:
    """
    Grants human approval for a queued action. Pops from the queue safely.

    [IMMUNEX-PATCH] Bug 8: logs if playbook.executable_actions ⊄ dqn_actions
    and sanitises the playbook report, but never mutates decision.actions.

    [IMMUNEX-PATCH] Bug 13: removed the pre-execution playbook pass that
    allowed the LLM's executable_actions to override the DQN's decision.
    The playbook is now generated AFTER execution, purely for documentation.

    [IMMUNEX-PATCH] Bug 9: uses .pop(alert_id, None) to prevent KeyError 500s
    under concurrent approvals.
    """
    # [IMMUNEX-PATCH] Bug 6: record arrival time immediately
    request_arrival_time = time.monotonic()

    # 404 check FIRST — before any body work
    if alert_id not in _pending_lookup:
        raise HTTPException(
            status_code = 404,
            detail      = f"No pending action found for alert_id={alert_id}",
        )

    # Parse optional JSON body
    req = ApprovalRequest(approved=True)        # safe default
    try:
        body_bytes = await raw_request.body()
        if body_bytes.strip():
            body_json = await raw_request.json()
            req = ApprovalRequest(**body_json)
    except Exception:
        pass    # empty / malformed body → keep default

    t_start = time.perf_counter()
    # [IMMUNEX-PATCH] Bug 9: safe pop — prevents KeyError 500 under race conditions
    pending_item = _pending_lookup.pop(alert_id, None)
    if pending_item is None:
        raise HTTPException(status_code=404, detail=f"alert_id={alert_id} was already processed")

    alert, decision, verification, reasoning, priority, ts, _arrival = pending_item

    if not req.approved:
        # Rejected via the approve endpoint (legacy path — prefer /admin/reject)
        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = False,
            approver = "human_operator",
            reason   = "Rejected via POST /approve endpoint",
        )
        audit_entry_id = _audit.log_incident(
            alert=alert, decision=decision, verification=verification,
            execution=None, playbook=None, modified_actions=req.modified_actions,
            priority=priority, before_actions=decision.actions.copy() if decision.actions else [],
            auto_approved=False, human_reviewed=True, approver_id="human_operator",
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
            audit_entry_id       = audit_entry_id,
        )

    before_actions = decision.actions.copy() if decision.actions else []
    dqn_actions    = set(decision.actions)

    # Apply human-modified actions if supplied
    if req.modified_actions:
        valid_ids = [idx for idx in req.modified_actions if idx in ACTION_NAMES]
        if valid_ids:
            decision.actions           = valid_ids
            decision.action_names      = [ACTION_NAMES.get(idx, "unknown") for idx in valid_ids]
            decision.action_categories = [get_action_category(idx) for idx in valid_ids]
            dqn_actions                = set(valid_ids)

    try:
        # Execute the DQN's approved actions directly — the playbook must
        # never override what the DQN decided (Bug 13 fix).
        execution = _executor.execute(decision, alert, approval_granted=True)

        # Generate the playbook from the REAL execution result.
        # The playbook is purely for documentation/audit — it does NOT
        # mutate decision.actions.
        playbook = _generator.generate(alert, decision, verification, execution)

        # [IMMUNEX-PATCH] Bug 8 (read-only): log if the playbook LLM
        # hallucinated action IDs outside the DQN's approved set, but
        # do NOT overwrite decision.actions — that caused Bug 13.
        if getattr(playbook, "executable_actions", None):
            hallucinated = set(playbook.executable_actions) - dqn_actions
            if hallucinated:
                logger.warning(
                    "playbook_hallucinated_actions_detected",
                    alert_id       = alert_id,
                    hallucinated   = list(hallucinated),
                    dqn_actions    = list(dqn_actions),
                )
                # Sanitise in-place on the report object so the audit trail
                # only records valid action IDs — but decision is untouched.
                playbook.executable_actions = [
                    a for a in playbook.executable_actions if a in dqn_actions
                ]

        # [IMMUNEX-PATCH] RLHF: insert into expert_demonstrations after approved execution
        await _db.insert_expert_demonstration(
            alert_id       = alert_id,
            state_vector   = alert.get("feature_vector", [0.0] * 128),
            expert_actions = decision.actions,
            admin_id       = "human_operator",
        )

        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = True,
            approver = "human_operator",
            reason   = "Manually approved via POST /approve endpoint",
        )
        # [IMMUNEX-PATCH] Bug 12: human_reviewed=True, auto_approved=False
        audit_entry_id = _audit.log_incident(
            alert=alert, decision=decision, verification=verification,
            execution=execution, playbook=playbook,
            modified_actions=req.modified_actions, priority=priority,
            before_actions=before_actions,
            auto_approved=False, human_reviewed=True, approver_id="human_operator",
        )

        logger.info(
            "approved_action_executed",
            alert_id       = alert_id,
            actions        = decision.action_names,
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


# ── [IMMUNEX-PATCH] New endpoint: POST /admin/reject ─────────────────────

@app.post(
    "/admin/reject",
    tags    = ["Approvals"],
    summary = "Reject a pending alert and store as RLHF negative example",
)
async def admin_reject(req: RejectRequest) -> dict:
    """
    Rejects a pending alert permanently.

    Workflow:
    1. Pop from _pending_lookup (safe pop, no KeyError).
    2. Insert state + DQN-proposed actions into rejected_demonstrations.
    3. Do NOT execute anything.
    4. Do NOT insert into expert_demonstrations.

    [IMMUNEX-PATCH] POST /admin/reject: stores the DQN's rejected proposal as
    a negative RLHF example so the rejection-memory lookup in response_engine.py
    (is_action_rejected) can prevent the model from re-proposing the same action
    in a near-identical state.
    """
    alert_id = req.alert_id

    # [IMMUNEX-PATCH] Bug 9: use .pop(alert_id, None) — no KeyError under races
    pending_item = _pending_lookup.pop(alert_id, None)
    if pending_item is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"No pending action found for alert_id={alert_id}",
        )

    alert, decision, verification, reasoning, priority, ts, _arrival = pending_item

    # Log rejection in audit trail
    if _audit:
        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = False,
            approver = req.admin_id,
            reason   = req.rejection_reason,
        )
        # [IMMUNEX-PATCH] Bug 12: human_reviewed=True for reject decisions
        _audit.log_incident(
            alert=alert, decision=decision, verification=verification,
            execution=None, playbook=None, priority=priority,
            auto_approved=False, human_reviewed=True, approver_id=req.admin_id,
        )

    # [IMMUNEX-PATCH] RLHF negative: insert into rejected_demonstrations
    await _db.insert_rejected_demonstration(
        alert_id         = alert_id,
        state_vector     = alert.get("feature_vector", [0.0] * 128),
        rejected_actions = decision.actions,
        rejection_reason = req.rejection_reason,
        admin_id         = req.admin_id,
    )

    logger.info(
        "alert_rejected_stored_as_negative_rlhf",
        alert_id         = alert_id,
        rejected_actions = decision.actions,
        admin_id         = req.admin_id,
        reason           = req.rejection_reason,
    )

    return {
        "status":   "rejected",
        "alert_id": alert_id,
        "actions_rejected": decision.actions,
        "rlhf_stored": True,
    }


# ── [IMMUNEX-PATCH] New endpoint: POST /admin/override ───────────────────

@app.post(
    "/admin/override",
    response_model = IncidentResponse,
    tags           = ["Approvals"],
    summary        = "Override DQN actions with human-selected actions (Z3 Pass 2)",
)
async def admin_override(req: OverrideRequest) -> IncidentResponse:
    """
    Human operator overrides the DQN's proposed actions with their own set.

    Workflow (Steps A–E):
    A. Z3 Pass 2: verify override_actions with is_human_reviewed=True.
       If Z3 denies → HTTP 400. (Human review unlocks C1 trading-window constraint.)
    B. LLM Pass 2: generate new reasoning for the overridden actions.
    C. Playbook Pass 2: generate new playbook for overridden actions.
    D. Execute via ActionExecutor.
    E. Insert into expert_demonstrations for offline RLHF retraining.

    [IMMUNEX-PATCH] POST /admin/override with is_human_reviewed=True:
    If a critical attack happens at 2 PM (trading hours), Z3 normally blocks
    quarantine_subnet (action 13). When the human admin hits "Override",
    is_human_reviewed=True unlocks C1 so the quarantine proceeds safely.
    Without this flag, Z3 would block the human's override too.
    """
    alert_id = req.alert_id
    request_arrival_time = time.monotonic()
    t_start  = time.perf_counter()

    # [IMMUNEX-PATCH] Bug 9: safe pop
    pending_item = _pending_lookup.pop(alert_id, None)
    if pending_item is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"No pending action found for alert_id={alert_id}",
        )

    alert, original_decision, original_verification, original_reasoning, priority, ts, _arrival = pending_item

    # Validate override action IDs against registry
    valid_override = [a for a in req.override_actions if a in ACTION_NAMES]
    if not valid_override:
        raise HTTPException(
            status_code = 400,
            detail      = f"No valid action IDs in override_actions. Valid range: 0–49. Got: {req.override_actions}",
        )

    # Build an ActionDecision for the override set
    override_decision = ActionDecision(
        alert_id          = alert_id,
        action_index      = valid_override[0],
        action_name       = ACTION_NAMES.get(valid_override[0], "unknown"),
        actions           = valid_override,
        action_names      = [ACTION_NAMES.get(a, "unknown") for a in valid_override],
        action_categories = [get_action_category(a) for a in valid_override],
        requires_approval = True,
        confidence        = original_decision.confidence,
        uncertain         = original_decision.uncertain,
        impact            = original_decision.impact,
        severity          = original_decision.severity,
        timestamp         = original_decision.timestamp,
        raw_q_values      = None,
    )

    # ── Step A: Z3 Pass 2 — verify override with is_human_reviewed=True ───
    # [IMMUNEX-PATCH] Bug 4 + Override requirement:
    # is_human_reviewed=True explicitly tells Z3's C1 constraint that a human
    # has physically confirmed this action. Without this, a quarantine_subnet
    # (action 13) during 2 PM trading hours would be blocked even on override.
    z3_pass2: VerificationResult = _verifier.verify(
        override_decision,
        alert,
        is_human_reviewed    = True,              # [IMMUNEX-PATCH] CRITICAL: human override flag
        request_arrival_time = request_arrival_time,  # [IMMUNEX-PATCH] Bug 6
    )

    if not z3_pass2.approved:
        logger.warning(
            "override_z3_pass2_denied",
            alert_id            = alert_id,
            override_actions    = valid_override,
            violated            = z3_pass2.violated_constraints,
            reason              = z3_pass2.reason,
        )
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"Z3 Pass 2 denied the override actions {valid_override}. "
                f"Violated: {z3_pass2.violated_constraints}. "
                f"Reason: {z3_pass2.reason}"
            ),
        )

    # ── Step B: LLM Pass 2 — generate reasoning for override actions ──────
    reasoning_target = ", ".join(override_decision.action_names)
    # [IMMUNEX-PATCH] Bug 12: human_reviewed=True so audit shows human intent
    reasoning_pass2 = await generate_llm_reasoning(
        alert,
        reasoning_target,
        z3_pass2.reason,
        auto_approved  = False,   # [IMMUNEX-PATCH] Bug 12
        human_reviewed = True,    # [IMMUNEX-PATCH] Bug 12: human drove this
    )

    # ── Step C: Playbook Pass 2 ────────────────────────────────────────────
    dummy_exec = ExecutionResult(
        alert_id=alert_id, action_index=override_decision.action_index,
        action_name=override_decision.action_name, actions=override_decision.actions,
        action_names=override_decision.action_names, status="simulated",
        dry_run=True, execution_time_ms=0.0, output={}, error=None,
        validation_status={}
    )
    playbook_pass2 = _generator.generate(alert, override_decision, z3_pass2, dummy_exec)

    # ── Step D: Execute ────────────────────────────────────────────────────
    execution = _executor.execute(override_decision, alert, approval_granted=True)

    # ── Step E: RLHF positive — insert override into expert_demonstrations ─
    # [IMMUNEX-PATCH] Override actions are stored as positive RLHF examples
    # so the DQN can learn to prefer human-validated actions in similar states.
    await _db.insert_expert_demonstration(
        alert_id       = alert_id,
        state_vector   = alert.get("feature_vector", [0.0] * 128),
        expert_actions = valid_override,
        admin_id       = req.admin_id,
    )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

    # Audit log
    if _audit:
        _audit.log_approval_decision(
            alert_id = alert_id,
            approved = True,
            approver = req.admin_id,
            reason   = f"Human override with actions {valid_override}",
        )
        # [IMMUNEX-PATCH] Bug 12: human_reviewed=True, auto_approved=False
        audit_entry_id = _audit.log_incident(
            alert=alert, decision=override_decision, verification=z3_pass2,
            execution=execution, playbook=playbook_pass2,
            before_actions=original_decision.actions, priority=priority,
            auto_approved=False, human_reviewed=True, approver_id=req.admin_id,
        )
    else:
        audit_entry_id = str(uuid.uuid4())

    logger.info(
        "admin_override_executed",
        alert_id        = alert_id,
        override_actions = valid_override,
        admin_id        = req.admin_id,
        duration_ms     = elapsed_ms,
    )

    return IncidentResponse(
        alert_id             = alert_id,
        status               = "responded",
        pipeline_duration_ms = elapsed_ms,
        decision             = asdict(override_decision),
        verification         = asdict(z3_pass2),
        execution            = asdict(execution),
        playbook             = asdict(playbook_pass2),
        llm_reasoning        = reasoning_pass2,
        audit_entry_id       = audit_entry_id,
        priority             = priority,
        uncertain            = getattr(override_decision, "uncertain", False),
        impact               = getattr(override_decision, "impact", "unknown"),
    )


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