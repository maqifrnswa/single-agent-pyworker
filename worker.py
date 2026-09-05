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
# within ~300s of starting ("timed out starting after 300s" in workergroup logs).
# That timeout is server-side (vast-ai autoscaler, types.cpp) — there is no 300s
# constant anywhere in the pyworker SDK and no workergroup knob for it; model
# load alone takes ~200s here, so the benchmark must finish in ~60-90s.
# BENCH_DEPTHS therefore samples 10k->50k (5 payloads: 1 warmup + 2 runs x 2).
# To cover the full curve to 100k once the budget allows, widen BENCH_DEPTHS
# toward (1, 3, 5, 7, 10) or raise runs — but re-check the 300s budget first.
NUM_TURNS = 10
BENCH_DEPTHS = (1, 2, 3, 4, 5)
USER_CHUNK_CHARS = 38000
ASSISTANT_ACK_CHARS = 2000
BENCHMARK_MAX_TOKENS = 128


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


def chat_workload(data) -> float:
    """Total input + output tokens (char/4 estimate), the Vast-recommended LLM cost proxy."""
    try:
        total_chars = 0
        messages = data.get("messages", [])
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    content = m.get("content", "")
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        # OpenAI structured content parts: [{type, text}, ...]
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                total_chars += len(part["text"])
        # Fall back for /v1/completions-style payloads sharing this calculator.
        prompt = data.get("prompt", "")
        if isinstance(prompt, str):
            total_chars += len(prompt)
        return float(data.get("max_tokens", 0) + total_chars / 4.0)
    except Exception:
        return float(data.get("max_tokens", 0))


class AgenticWorkflowGenerator:
    """Growing multi-turn chain: depth d ~= d * 10k tokens.

    - Turn chunks are prebuilt once with fixed seeds, so depth d always equals
      chunks[0..d] byte-identically. Concurrent benchmark requests at different
      depths still share exact prefixes -> vLLM prefix cache reuses KV blocks
      and only prefills the ~10k delta, matching prod (10k in, +10k/turn).
    - Depths cycle through BENCH_DEPTHS so one startup benchmark covers a
      representative slice of the 10k->100k curve instead of a single point.
      Kept to 5 payloads (1 warmup + 2 runs x 2 concurrent) to fit Vast's
      ~300s server-side starting budget (see constants above).
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
        return {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
        }


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
                workload_calculator=lambda data: float(data.get("max_tokens", 0)),
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
                                    generator=agentic_workflow_generator, concurrency=2, runs=3
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
        load_log_msgs=["Application startup complete."],
        #error_log_msgs=["INFO exited: vllm", "RuntimeError: Engine", "Traceback (most recent call last):"],
        error_log_msgs=["INFO exited: vllm", "Traceback (most recent call last):"],
    ))