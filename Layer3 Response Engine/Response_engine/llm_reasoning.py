"""
IMMUNEX — Layer 3: LLM Reasoning and Human-in-the-Loop decision layer
=======================================================================
Calls a locally-hosted Llama-3 model (via Ollama) to produce a risk
assessment and rationale for every proposed action before execution.

If Ollama is unavailable or times out (8 s), returns a safe fallback
dict so the pipeline never stalls.

Human approval behaviour
------------------------
  IMMUNEX_CLI_MODE=true   → interactive prompt with 10 s timeout
                            (auto-approves on timeout / EOF)
  IMMUNEX_CLI_MODE=false  → auto-approves immediately
                            (correct for API / server mode)

Test input override
-------------------
  IMMUNEX_TEST_INPUT_FILE → path to a file containing "y" or "n";
                            used by automated test suites to inject
                            a deterministic human decision without
                            blocking on stdin.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import structlog
import ollama as ollama_client

logger = structlog.get_logger("immunex.llm_reasoning")


# ── Internal Ollama call (blocking, runs in thread pool) ──────────────────

def _call_ollama_sync(prompt: str) -> str | None:
    """
    Sends *prompt* to the local Ollama server and returns the raw response
    string.  Returns None on any error so callers always get a safe value.
    """
    try:
        host   = os.environ.get("IMMUNEX_OLLAMA_HOST", "http://localhost:11434")
        model  = os.environ.get("IMMUNEX_OLLAMA_MODEL", "llama3.1:8b")
        client = ollama_client.Client(host=host)
        response = client.chat(
            model    = model,
            messages = [
                {
                    "role":    "system",
                    "content": (
                        "You are a cybersecurity expert. "
                        "Provide structured JSON only. No markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.get("message", {}).get("content")
    except Exception as exc:
        logger.warning("ollama_llm_call_failed", error=str(exc))
        return None


# ── Public async API ───────────────────────────────────────────────────────

async def generate_llm_reasoning(
    state:      dict,
    action_name: str,
    z3_result:  str,
) -> dict:
    """
    Calls Ollama to generate a risk explanation for the proposed action.

    Never blocks indefinitely and never propagates exceptions.
    Returns a safe fallback dict on any failure.

    Returns
    -------
    dict with keys: reason, risk (LOW/MEDIUM/HIGH), business_impact,
    alternative (action name string for the fallback action).
    """
    prompt = f"""You are a cybersecurity expert.
Given:
- Action: {action_name}
- Safety Check: {z3_result}
- Context: network intrusion detection in a banking environment

Explain:
1. Why this action is appropriate
2. Risk level (LOW, MEDIUM, or HIGH)
3. Business impact
4. Safer alternative if risk is high (use an action name from the IMMUNEX registry,
   e.g. do_nothing, increase_log_verbosity, flag_for_human_review)

Respond ONLY in JSON with EXACT keys: "reason", "risk", "business_impact", "alternative"."""

    _fallback: dict = {
        "reason":          "LLM unavailable — proceeding with default risk classification",
        "risk":            "MEDIUM",
        "business_impact": "UNKNOWN",
        "alternative":     "do_nothing",
    }

    loop     = asyncio.get_running_loop()
    raw_resp = None

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            raw_resp = await asyncio.wait_for(
                loop.run_in_executor(pool, _call_ollama_sync, prompt),
                timeout=8.0,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("llm_reasoning_timeout_or_error", error=str(exc))
            raw_resp = None

    if not raw_resp:
        return _fallback

    # ── Parse JSON response (3-stage, same strategy as PlaybookGenerator) ─
    try:
        # Stage 1: direct parse
        try:
            parsed = json.loads(raw_resp)
        except json.JSONDecodeError:
            # Stage 2: strip markdown fences
            stripped = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.MULTILINE)
            stripped = re.sub(r"```\s*$",           "", stripped.strip(), flags=re.MULTILINE)
            try:
                parsed = json.loads(stripped.strip())
            except json.JSONDecodeError:
                # Stage 3: extract first {...} block
                match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                else:
                    raise ValueError("No JSON object found in LLM response")

        # Ensure all expected keys are present — fill missing ones from fallback
        for key, default_val in _fallback.items():
            if key not in parsed:
                parsed[key] = default_val

        return parsed

    except Exception as exc:
        logger.warning("llm_reasoning_parse_failed", error=str(exc))
        return _fallback


async def human_approval(action_name: str, reasoning: dict) -> bool:
    """
    Requests human approval for the proposed action.

    Behaviour is controlled by the IMMUNEX_CLI_MODE environment variable:

    - CLI mode  (true) : prints an interactive prompt and waits up to 10 s.
                         Auto-approves on timeout or EOF.
    - API mode (false) : auto-approves immediately without blocking the
                         server event loop (correct default for uvicorn).

    Returns True (approved) or False (rejected).
    """
    is_cli = os.environ.get("IMMUNEX_CLI_MODE", "false").lower() == "true"

    if not is_cli:
        # API mode: never block the event loop
        return True

    def _ask() -> str:
        # Allow automated tests to inject a decision via a file
        test_file = os.environ.get("IMMUNEX_TEST_INPUT_FILE")
        if test_file and os.path.exists(test_file):
            with open(test_file, "r") as fh:
                content = fh.read().strip()
            return "n" if content == "n" else "y"

        try:
            return input(
                f"\n[HITL] Approve action '{action_name}'? "
                f"[Risk: {reasoning.get('risk', 'UNKNOWN')}] (y/n): "
            )
        except (EOFError, OSError):
            return "y"

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            ans = await asyncio.wait_for(
                loop.run_in_executor(pool, _ask),
                timeout=10.0,
            )
            return ans.strip().lower().startswith("y")
        except (asyncio.TimeoutError, Exception):
            print("\n[HITL] Auto-approved due to timeout or error.")
            return True