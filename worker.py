"""Shared core for the OpenAI-compatible workers (vllm/sglang/llama/openai).

They all proxy the same /v1/completions + /v1/chat/completions API, so the logic lives
here and the per-engine adapters just pass an EngineDefaults. Every default is
env-overridable: the image is version-locked to the engine, so it owns the
engine/version-specific values (log path, health endpoint, log grammar)."""

import os
import random
import string
import threading
from dataclasses import dataclass, field
from typing import List

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig


def _env_lines(name, default):
    """Newline-delimited env var -> list of stripped lines; default if unset/empty."""
    raw = os.environ.get(name)
    return [s for ln in raw.splitlines() if (s := ln.strip())] if raw else default


# One template serves both lanes; on-demand templates set only the engine var, so the
# benchmark recovers the model id from it. Only one is ever set.
_MODEL_NAME_VARS = ("MODEL_NAME", "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL")


def _resolve_model_name():
    return next((v for var in _MODEL_NAME_VARS if (v := os.environ.get(var))), None)


def _resolve_model_name_or_raise():
    model = _resolve_model_name()
    if not model:
        raise ValueError(
            "No model set: MODEL_NAME / VLLM_MODEL / SGLANG_MODEL / LLAMA_MODEL all empty"
        )
    return model


@dataclass(frozen=True)
class EngineDefaults:
    """Per-engine baked defaults; each is overridden by the matching env var if set."""

    name: str                 # engine id for the startup banner
    model_log_file: str       # MODEL_LOG
    load_log_msgs: List[str]  # MODEL_LOAD_LOG_MSG — model-loaded markers
    error_log_msgs: List[str]  # MODEL_ERROR_LOG_MSGS — failed-load markers
    info_log_msgs: List[str] = field(default_factory=lambda: ['"message":"Download'])  # MODEL_INFO_LOG_MSGS


MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000

# Benchmark shape: start ~10k tokens, +~10k per turn.
# 1 token ~= 4 chars. Each turn ~= 1 user chunk (~9.5k tok) + 1 short ack (~0.5k tok).
#
# STARTUP BUDGET: Vast's control plane marks a worker error if it is not ready
# within ~300s of starting ("timed out starting after 300s" in workergroup logs)
# and ~800-1000s in loading ("timed out loading after ...s": 792s and 1012s
# observed on different workers). That timeout is server-side (vast-ai
# autoscaler, types.cpp) — there is no timeout constant anywhere in the
# pyworker SDK and no workergroup knob for it.
# BENCHMARK DESIGN (steady-state, not curve slice): the SDK runs exactly one
# unmeasured warmup payload, then runs x concurrency measured payloads, and
# reports the max. All payloads are full depth-4 with byte-identical
# deterministic prefixes, so the single warmup pays the one cold prefill and
# every measured run is prefix-cache hits + 512-token decode: the prod steady
# state (warm ~100k-token context, 250-500 token outputs; ignore_eos forces
# the full 512 for deterministic decode load). runs=2 is safe because no
# measured run is cold.
# TOKEN DENSITY WARNING: the seeded filler (random lower/digit words) tokenizes
# at ~1.5 chars/token, not the ~4 of natural text — measured live 2026-09-06:
# depth-10 (399k chars) tokenized to 261633 input tokens and overflowed the
# 262144 limit by one token (400 BadRequestError). Depth-4 (~159k chars) is
# ~105k engine tokens ~= prod 100k context. If the filler ever changes,
# re-measure true token counts before trusting depth numbers. Slow hosts
# (~250 tok/s effective prefill) may still time out warming 105k inside the
# loading window; they cannot serve this workload, so failing loudly beats a
# misleading score.
NUM_TURNS = 10
BENCH_DEPTHS = (4, 4, 4)
USER_CHUNK_CHARS = 38000
ASSISTANT_ACK_CHARS = 2000
BENCHMARK_MAX_TOKENS = 512  # this must be typical, average for workload
# Average output tokens charged per request. Measured on real opencode agentic
# traffic (text+reasoning): mean ~471, median 166, p90 ~1457. Env-overridable.
AVERAGE_OUTPUT_TOKENS = float(os.environ.get("WORKLOAD_AVG_OUTPUT_TOKENS",
                                             str(BENCHMARK_MAX_TOKENS)))
# Fraction of prompt tokens charged as uncached prefill. Real agentic traffic
# measured ~86% prefix-cache hits (=> ~0.14 uncached); 0.25 is a conservative
# upper bound. Env-overridable.
WORKLOAD_UNCACHED_FRACTION = float(os.environ.get("WORKLOAD_UNCACHED_FRACTION", "0.25"))
# Fallback chars-per-token when the tokenizer is unavailable (~4 natural text,
# ~3.5 agentic code/JSON). Env-overridable.
WORKLOAD_CHARS_PER_TOKEN = float(os.environ.get("WORKLOAD_CHARS_PER_TOKEN", "4.0"))

# Optional tool-call shape for benchmark payloads. The benchmark only measures
# throughput, so the model need not actually call tools - but a real tool schema
# makes input token density match production. Off by default: a server without
# tool support would 400. Enable with BENCH_TOOL_CALLS=1.
BENCH_TOOL_CALLS = os.environ.get("BENCH_TOOL_CALLS", "0") == "1"
BENCH_TOOLS = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "workdir": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}]

def _seeded_chunk_chars(seed: int, n_chars: int) -> str:
    """Deterministic diverse filler: same seed always yields the same string.

    Uses a seeded RNG so every worker (and every benchmark call) reproduces
    identical prefixes. Prefix-cache hits require byte-identical token prefixes,
    so determinism is load-bearing. Deliberately varied (not `"..." * N`) so
    prefill cost resembles real diverse context.
    """
    rng = random.Random(seed)
    alphabet = string.ascii_lowercase + string.digits
    out = []
    remaining = n_chars
    while remaining > 0:
        word_len = rng.randint(3, 10)
        word = "".join(rng.choice(alphabet) for _ in range(word_len))
        out.append(word)
        remaining -= word_len + 1  # +1 for the space
    return " ".join(out)[:n_chars]


def request_parser(request):
    return request["input"] if request.get("input") is not None else request


_TOKENIZER = None
_TOKENIZER_TRIED = False


def _tokenizer():
    """Lazy model tokenizer from the served model dir; None if unavailable."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        try:
            from tokenizers import Tokenizer  # provided by requirements.txt
            model_dir = _resolve_model_name_or_raise()
            _TOKENIZER = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        except Exception as e:
            print(f"workload: tokenizer unavailable ({type(e).__name__}: {e}); "
                  f"falling back to {WORKLOAD_CHARS_PER_TOKEN} chars/token", flush=True)
            _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> float:
    """Exact token count when a tokenizer is available, else a char estimate."""
    if not text:
        return 0.0
    tok = _tokenizer()
    if tok is not None:
        try:
            return float(len(tok.encode(text, add_special_tokens=False).ids))
        except Exception:
            pass
    return len(text) / WORKLOAD_CHARS_PER_TOKEN


def _prompt_text(data) -> str:
    """All input text from a chat or completions payload."""
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
    return "\n".join(parts)


def expected_output_tokens(data) -> float:
    """Charged output tokens: the client cap only when it is BELOW the measured
    average (a cap is an upper bound, never an expectation), else the average."""
    mt = data.get("max_tokens")
    if mt is None:
        return AVERAGE_OUTPUT_TOKENS
    try:
        return min(float(mt), AVERAGE_OUTPUT_TOKENS)
    except (TypeError, ValueError):
        return AVERAGE_OUTPUT_TOKENS


def chat_workload(data) -> float:
    """Charged workload = expected output + uncached_fraction * prompt tokens.

    Uses the real model tokenizer when available (chars/token fallback), so dense
    code/JSON/tool-schema prompts are not under-counted by a fixed chars/4.
    Works for both chat and completions payloads."""
    try:
        return (expected_output_tokens(data)
                + WORKLOAD_UNCACHED_FRACTION * count_tokens(_prompt_text(data)))
    except Exception:
        return float(data.get("max_tokens", 0))


class AgenticWorkflowGenerator:
    """Growing multi-turn chain: depth d ~= d * 10k tokens.

    - Turn chunks are prebuilt once with fixed seeds, so depth d always equals
      chunks[0..d] byte-identically. Concurrent benchmark requests at the same
      depth share exact prefixes -> vLLM prefix cache reuses KV blocks.
    - Steady-state design: BENCH_DEPTHS is all depth-10 (~100k tokens), so the
      single SDK warmup pays the one cold 100k prefill and every measured run
      is prefix-cache hits + full decode — the prod steady state.
    - Thread-safe counter for concurrent payload generation.
    """

    def __init__(self, depths=(1, 2, 3, 4, 5), num_turns: int = NUM_TURNS, max_tokens: int = BENCHMARK_MAX_TOKENS):
        self.depths = tuple(depths)
        self.num_turns = num_turns
        self.max_tokens = max_tokens
        self.system_message = (
            "You are an autonomous AI agent performing multi-step reasoning. "
            "Use the conversation history to execute the next step."
        )
        self.user_chunks = [
            f"Step {i + 1} context observations: {_seeded_chunk_chars(1000 + i, USER_CHUNK_CHARS)}"
            for i in range(num_turns)
        ]
        self.assistant_acks = [
            f"Step {i + 1} result: {_seeded_chunk_chars(2000 + i, ASSISTANT_ACK_CHARS)}"
            for i in range(num_turns)
        ]
        self._counter = 0
        self._lock = threading.Lock()

    def __call__(self) -> dict:
        model = _resolve_model_name_or_raise()
        with self._lock:
            depth = self.depths[self._counter % len(self.depths)]
            self._counter += 1
        messages = [{"role": "system", "content": self.system_message}]
        for i in range(depth):
            messages.append(
                {"role": "user", "content": f"Execute agentic step {i + 1}.\n{self.user_chunks[i]}"}
            )
            if i < depth - 1:
                messages.append({"role": "assistant", "content": self.assistant_acks[i]})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "ignore_eos": True,
        }
        if BENCH_TOOL_CALLS:
            payload["tools"] = BENCH_TOOLS
            payload["tool_choice"] = "auto"
        return payload


def run(defaults: EngineDefaults) -> None:
    """Build the WorkerConfig from defaults (env-overridable) and run the worker."""

    # BACKEND=openai aliases vllm, so report the real engine and note the alias.
    backend = os.environ.get("BACKEND")
    alias = f" (BACKEND={backend})" if backend and backend != defaults.name else ""
    print(f"Using worker backend: {defaults.name}{alias}", flush=True)

    agentic_workflow_generator = AgenticWorkflowGenerator(depths=BENCH_DEPTHS)

    # Relative path resolves against the server url+port; a full URL is used as-is.
    healthcheck_url = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")

    config = dict(
        model_server_url=MODEL_SERVER_URL,
        model_server_port=MODEL_SERVER_PORT,
        model_log_file=os.environ.get("MODEL_LOG", defaults.model_log_file),
        model_healthcheck_url=healthcheck_url,
        handlers=[
            HandlerConfig(
                route="/v1/completions",
                workload_calculator=chat_workload,   # charges input too (was max_tokens-only)
                allow_parallel_requests=True,
                request_parser=request_parser,
                max_queue_time=600.0
                ),
            HandlerConfig(
                route="/v1/chat/completions",
                workload_calculator=chat_workload,
                allow_parallel_requests=True,
                request_parser=request_parser,
                max_queue_time=600.0,
                benchmark_config=BenchmarkConfig(
                                    generator=agentic_workflow_generator, concurrency=2, runs=2
                                ),
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=_env_lines("MODEL_LOAD_LOG_MSG", defaults.load_log_msgs),
            on_error=_env_lines("MODEL_ERROR_LOG_MSGS", defaults.error_log_msgs),
            on_info=_env_lines("MODEL_INFO_LOG_MSGS", defaults.info_log_msgs),
        ),
    )
    Worker(WorkerConfig(**config)).run()


if __name__ == "__main__":
    # run it
    run(EngineDefaults(
        name="vllm",
        model_log_file="/var/log/portal/vllm.log",
        load_log_msgs=["Application startup complete.", "engines connected"],
        # Error patterns deliberately minimal: process death is the only
        # load-bearing signal. Bare "Traceback" false-positives on benign
        # torch inductor cubin-cache warnings (W0906, engine healthy), and
        # "RuntimeError: Engine" fires on transient allocator OOMs the engine
        # survives. All true fatals observed (EngineCore deaths) end with the
        # supervisor reporting vllm EXITED, which "INFO exited: vllm" catches.
        # Override per-template via MODEL_ERROR_LOG_MSGS (newline-delimited).
        error_log_msgs=["INFO exited: vllm", "EngineCore failed to start"],
    ))
