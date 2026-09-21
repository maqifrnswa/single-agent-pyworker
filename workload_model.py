"""Shared workload model: W_PREFILL · fresh + W_CACHED · cached + output, with fixed hit_rate. Pure stdlib + optional tokenizer plug."""

import hashlib
import os
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Constants (env-overridable)
# --------------------------------------------------------------------------- #
# Weights for the three resource components (fixed, ora-2; see RESEARCH.md:
# "cached prefill ~= 1/20 the cost of fresh prefill", measured steady-state
# cache hit rate ~86%). With hit_rate = 0.86 the formula collapses to
# output + 0.183 * prompt_tokens, but we keep the decomposed form for clarity.
W_PREFILL = float(os.environ.get("WORKLOAD_W_PREFILL", "1.0"))
W_CACHED = float(os.environ.get("WORKLOAD_W_CACHED", "0.05"))
DEFAULT_HIT_RATE = float(os.environ.get("WORKLOAD_DEFAULT_HIT_RATE", "0.86"))

# Fallback chars-per-token when no tokenizer is registered (~4 natural text,
# ~3.5 agentic code/JSON). Env-overridable.
WORKLOAD_CHARS_PER_TOKEN = float(os.environ.get("WORKLOAD_CHARS_PER_TOKEN", "4.0"))

# Backward-compat alias (the harness imports CHARS_PER_TOKEN as _WM_CHARS_PER_TOKEN).
CHARS_PER_TOKEN = WORKLOAD_CHARS_PER_TOKEN

# Average output tokens charged per request. Measured on real opencode agentic
# traffic (text+reasoning): mean ~471, median 166, p90 ~1457. Env-overridable.
AVERAGE_OUTPUT_TOKENS = float(os.environ.get("WORKLOAD_AVG_OUTPUT_TOKENS", "512"))


# --------------------------------------------------------------------------- #
# Tokenizer (pluggable) — worker wires up the real one; harness leaves None.
# --------------------------------------------------------------------------- #
_TOKENIZER: Optional[Callable[[str], int]] = None


def set_tokenizer(tokenizer_fn: Optional[Callable[[str], int]]) -> None:
    """Register a tokenizer callable: text -> token count (int)."""
    global _TOKENIZER
    _TOKENIZER = tokenizer_fn


def count_tokens(text: str) -> float:
    """Token count using the registered tokenizer, or char-based fallback.

    Returns a float (not int) for consistency with the char-based estimate.
    """
    if not text:
        return 0.0
    if _TOKENIZER is not None:
        try:
            return float(_TOKENIZER(text))
        except Exception:
            pass
    return len(text) / WORKLOAD_CHARS_PER_TOKEN


def prefix_hash(prompt_text: str) -> str:
    """Stable sha256[:16] of the full prompt text (log lines / debugging)."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Charged output tokens
# --------------------------------------------------------------------------- #
def expected_output_tokens(data: dict, default: float = AVERAGE_OUTPUT_TOKENS,
                           cap_only_below: bool = True) -> float:
    """Charged output tokens: the client cap only when it is BELOW the average.

    A cap is an upper bound (never an expectation), so we take min(max_tokens, average).
    If max_tokens is absent or invalid, return the average.
    """
    mt = data.get("max_tokens")
    if mt is None:
        return default
    try:
        mt_float = float(mt)
        if cap_only_below:
            return min(mt_float, default)
        else:
            return mt_float
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Core workload formula
# --------------------------------------------------------------------------- #
def _extract_prompt_tokens(data: dict) -> float:
    """Extract prompt text from a chat or completions payload and count tokens.

    Handles both:
      - chat: {"messages": [{"role": "system", "content": "..."}, ...]}
      - completions: {"prompt": "..."}
    """
    parts = []
    messages = data.get("messages", [])
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict):
                content = m.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            parts.append(part["text"])
    prompt = data.get("prompt", "")
    if isinstance(prompt, str):
        parts.append(prompt)
    return count_tokens("\n".join(parts))


def charge_workload(data: dict, *, prompt_tokens: Optional[int] = None,
                    hit_rate: float = DEFAULT_HIT_RATE) -> float:
    """Compute the charged workload for a request.

    Formula:
      work = expected_output_tokens(data)
           + W_PREFILL * prompt_tokens * (1 - hit_rate)
           + W_CACHED  * prompt_tokens * hit_rate

    Args:
      data: the request payload (dict with "messages" or "prompt", "max_tokens")
      prompt_tokens: explicit prompt token count (None = count_tokens on the payload)
      hit_rate: fraction of prompt tokens that are prefix-cache hits
                (fixed default; no EWMA tracking in this redesign)

    Returns:
      The charged workload (float, in "decode-equivalent seconds" units).
    """
    out = expected_output_tokens(data)
    p = prompt_tokens if prompt_tokens is not None else _extract_prompt_tokens(data)
    return out + W_PREFILL * p * (1 - hit_rate) + W_CACHED * p * hit_rate