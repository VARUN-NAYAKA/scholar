"""
ScholAR - Shared LLM utility.
Uses google-genai SDK with gemini-2.5-flash.
Includes rate limiting to avoid burning through free tier quota.
"""

import os
import time
import threading
from google import genai
from google.genai import types
from rich.console import Console

console = Console()

MODEL = "gemini-2.5-flash"

# ── Rate limiter: max 10 requests per minute (free tier = 10 RPM) ──
_lock = threading.Lock()
_call_times: list[float] = []
_MAX_RPM = 8  # stay under the 10 RPM limit


def _rate_limit():
    """Ensure we don't exceed RPM quota."""
    with _lock:
        now = time.time()
        # Remove calls older than 60s
        _call_times[:] = [t for t in _call_times if now - t < 60]
        if len(_call_times) >= _MAX_RPM:
            wait = 60 - (now - _call_times[0]) + 1
            console.print(f"[yellow]Rate limiter: waiting {wait:.0f}s to stay under quota...[/yellow]")
            time.sleep(wait)
        _call_times.append(time.time())


def call_gemini(prompt: str, max_tokens: int = 4096, temperature: float = 0.3, max_retries: int = 3) -> str:
    """
    Call Gemini 2.5 Flash with built-in rate limiting and retry.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY set in environment")

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        _rate_limit()
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err_msg = str(e)
            is_quota = "429" in err_msg or "quota" in err_msg.lower() or "RESOURCE_EXHAUSTED" in err_msg

            if is_quota and attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)
                console.print(f"[yellow]Rate limited (attempt {attempt+1}), waiting {wait_time}s...[/yellow]")
                time.sleep(wait_time)
            else:
                raise
