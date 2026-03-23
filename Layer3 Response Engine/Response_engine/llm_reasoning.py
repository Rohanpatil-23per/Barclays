"""
LLM Reasoning and Human-in-the-Loop decision layer.
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import structlog
import ollama as ollama_client

logger = structlog.get_logger("immunex.llm_reasoning")


def _call_ollama_sync(prompt: str) -> str | None:
    try:
        host = os.environ.get("IMMUNEX_OLLAMA_HOST", "http://localhost:11434")
        client = ollama_client.Client(host=host)
        response = client.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cybersecurity expert. Provide structured JSON. No markdown."
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response.get("message", {}).get("content")
    except Exception as exc:
        logger.warning("ollama_llm_call_failed", error=str(exc))
        return None


async def generate_llm_reasoning(state: dict, action_name: str, z3_result: str) -> dict:
    """
    Calls Ollama to generate an explanation and risk analysis for the proposed action.
    Never blocks or propagates exceptions. Returns a fallback dict on any failure.
    """
    prompt = f"""You are a cybersecurity expert.
Given:
- Action: {action_name}
- Safety Check: {z3_result}
- Context: network intrusion detection

Explain:
1. Why this action is appropriate
2. Risk level (LOW, MEDIUM, or HIGH)
3. Business impact
4. Safer alternative if risk is high (e.g., do_nothing)

Respond ONLY in JSON with EXACT keys: "reason", "risk", "business_impact", "alternative"."""

    fallback = {
        "reason": "LLM unavailable",
        "risk": "MEDIUM",
        "business_impact": "UNKNOWN",
        "alternative": "do_nothing"
    }

    loop = asyncio.get_running_loop()
    raw_resp = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            # 8-second timeout for the LLM call to prevent stalls
            raw_resp = await asyncio.wait_for(
                loop.run_in_executor(pool, _call_ollama_sync, prompt),
                timeout=8.0
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("llm_reasoning_timeout_or_error", error=str(exc))
            raw_resp = None

    if not raw_resp:
        return fallback

    try:
        import re
        stripped = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.MULTILINE)
        stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = json.loads(stripped)

        # Ensure all expected keys exist
        for key in fallback.keys():
            if key not in parsed:
                parsed[key] = fallback[key]

        return parsed

    except Exception as exc:
        logger.warning("llm_reasoning_parse_failed", error=str(exc))
        return fallback


async def human_approval(action_name: str, reasoning: dict) -> bool:
    """
    Asks for human approval.
    If IMMUNEX_CLI_MODE is true, asks interactively with a 10s timeout (auto-approves on timeout/EOF).
    Otherwise (API mode), auto-approves immediately to avoid blocking the server loop.
    """
    is_cli = os.environ.get("IMMUNEX_CLI_MODE", "false").lower() == "true"
    if not is_cli:
        return True

    def _ask() -> str:
        test_file = os.environ.get("IMMUNEX_TEST_INPUT_FILE")
        if test_file and os.path.exists(test_file):
            with open(test_file, "r") as f:
                content = f.read().strip()
            # For testing: if file says 'n', return 'n'. Otherwise 'y'.
            return "n" if content == "n" else "y"
            
        try:
            return input(f"\n[HITL] Approve action '{action_name}'? [Risk: {reasoning.get('risk')}] (y/n): ")
        except (EOFError, OSError):
            return "y"

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            # 10 second timeout for human input
            ans = await asyncio.wait_for(
                loop.run_in_executor(pool, _ask),
                timeout=10.0
            )
            return ans.strip().lower().startswith('y')
        except (asyncio.TimeoutError, Exception):
            print("\n[HITL] Auto-approved due to timeout or error.")
            return True
            
    return True
