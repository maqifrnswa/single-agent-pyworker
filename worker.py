"""Shared core for the OpenAI-compatible workers (vllm/sglang/llama/openai).

They all proxy the same /v1/completions + /v1/chat/completions API, so the logic lives
here and the per-engine adapters just pass an EngineDefaults. Every default is
env-overridable: the image is version-locked to the engine, so it owns the
engine/version-specific values (log path, health endpoint, log grammar)."""

import hashlib
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

# STARTUP BUDGET: Vast's control plane marks a worker error if it is not ready
# within ~300s of starting ("timed out starting after 300s" in workergroup logs)
# and ~800-1000s in loading ("timed out loading after ...s": 792s and 1012s
# observed on different workers). That timeout is server-side (vast-ai
# autoscaler, types.cpp) — there is no timeout constant anywhere in the
# pyworker SDK and no workergroup knob for it.
# BENCHMARK DESIGN (steady-state, not curve slice): every measured request is a
# byte-identical shared prefix (a prefix-cache hit after the SDK's single
# unmeasured warmup) plus a unique fresh tail (real prefill), and a forced full
# decode (ignore_eos). The shape is IDENTICAL for every request and every run,
# so the SDK's max-over-runs cannot select a warmer/faster/unrepresentative run.
# Payloads are sized in REAL TOKENS via count_tokens(), and the benchmark's
# forced decode length equals the calculator's charged output term, so charge
# and measurement agree. runs=2 is safe because no measured run is cold.
# TOKEN DENSITY NOTE: the seeded filler (random lower/digit words) tokenizes at
# ~1.5 chars/token, not the ~4 of natural text; sizing goes through
# count_tokens(), so this only affects byte length, not the token geometry.
# Slow hosts (~250 tok/s effective prefill) may still time out warming the
# prefix inside the loading window; they cannot serve this workload, so failing
# loudly beats a misleading score.
# Average output tokens charged per request. Measured on real opencode agentic
# traffic (text+reasoning): mean ~471, median 166, p90 ~1457. Env-overridable.
AVERAGE_OUTPUT_TOKENS = float(os.environ.get("WORKLOAD_AVG_OUTPUT_TOKENS", "512"))
# Fraction of prompt tokens charged as uncached prefill. Real agentic traffic
# measured ~86% prefix-cache hits (=> ~0.14 uncached); 0.25 is a conservative
# upper bound. Env-overridable.
WORKLOAD_UNCACHED_FRACTION = float(os.environ.get("WORKLOAD_UNCACHED_FRACTION", "0.25"))
# Fallback chars-per-token when the tokenizer is unavailable (~4 natural text,
# ~3.5 agentic code/JSON). Env-overridable.
WORKLOAD_CHARS_PER_TOKEN = float(os.environ.get("WORKLOAD_CHARS_PER_TOKEN", "4.0"))

# Benchmark geometry, in REAL TOKENS (production-shaped). Every measured request
# is: byte-identical shared prefix (a prefix-cache hit) + a unique fresh tail
# (real prefill), plus a forced full decode. Shape is identical for every
# request and run, so max-over-runs cannot pick a warmer/faster run.
BENCH_PREFIX_TOKENS = int(os.environ.get("BENCH_PREFIX_TOKENS", "61060"))  # shared, cached
BENCH_TAIL_TOKENS = int(os.environ.get("BENCH_TAIL_TOKENS", "9940"))       # fresh, unique per request
BENCH_MAX_MODEL_LEN = int(os.environ.get("BENCH_MAX_MODEL_LEN", "262144"))
BENCH_PREFIX_SEED = int(os.environ.get("BENCH_PREFIX_SEED", "1000"))
# The calculator's output term and the benchmark's forced decode length MUST be
# the same number so charge and measurement agree.
BENCHMARK_MAX_TOKENS = int(AVERAGE_OUTPUT_TOKENS)

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


def _seeded_text_for_tokens(seed: int, target_tokens: int) -> str:
    """Deterministic filler sized to ~target_tokens using count_tokens().

    Bisects the character count so the SAME estimator used for the workload
    charge also sizes the payload (consistent with or without a tokenizer).
    """
    target_tokens = max(1, int(target_tokens))
    lo, hi = 1, max(64, target_tokens * 8)
    best = _seeded_chunk_chars(seed, lo)
    for _ in range(48):
        mid = (lo + hi) // 2
        text = _seeded_chunk_chars(seed, mid)
        n = count_tokens(text)
        best = text
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid - 1
        if abs(n - target_tokens) <= max(2, 0.005 * target_tokens):
            break
    return best


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
    """Production-shaped benchmark payload: shared cached prefix + unique fresh tail.

    Each request is `system + one user message` where the user content is
    `shared_prefix + "\\n" + tail_k`. `shared_prefix` is byte-identical for every
    request (so it is a prefix-cache hit after the first request); `tail_k` is
    unique per request index (so every request pays real prefill). Both are
    deterministic, so repeated runs have identical shape.

    Construction FAILS LOUDLY if the payload cannot be sized or would not fit the
    model's context - a mis-sized benchmark must never be scored silently.
    """

    def __init__(self, prefix_tokens=BENCH_PREFIX_TOKENS, tail_tokens=BENCH_TAIL_TOKENS,
                 max_tokens=BENCHMARK_MAX_TOKENS, max_model_len=BENCH_MAX_MODEL_LEN,
                 prefix_seed=BENCH_PREFIX_SEED):
        self.prefix_tokens = int(prefix_tokens)
        self.tail_tokens = int(tail_tokens)
        self.max_tokens = int(max_tokens)
        self.max_model_len = int(max_model_len)
        self.system_message = (
            "You are an autonomous AI agent performing multi-step reasoning. "
            "Use the conversation history to execute the next step."
        )
        self.shared_prefix = _seeded_text_for_tokens(prefix_seed, self.prefix_tokens)
        self._tails = {}
        self._counter = 0
        self._lock = threading.Lock()

        prompt_tokens = count_tokens(self.shared_prefix) + count_tokens(self.system_message)
        drift = abs(prompt_tokens - self.prefix_tokens)
        assert drift <= max(4, 0.02 * self.prefix_tokens), (
            f"bench prefix sizing failed: {prompt_tokens:.0f} tok vs target "
            f"{self.prefix_tokens}")
        assert prompt_tokens + self.tail_tokens + self.max_tokens <= self.max_model_len, (
            f"bench payload {prompt_tokens + self.tail_tokens:.0f} + {self.max_tokens} out "
            f"exceeds max_model_len {self.max_model_len}")
        self.prefix_hash = hashlib.sha256(self.shared_prefix.encode()).hexdigest()[:16]
        has_tok = _tokenizer() is not None
        print(f"bench: prefix={prompt_tokens:.0f} tok (sha256={self.prefix_hash}, "
              f"exact_tokenizer={has_tok}), tail={self.tail_tokens} tok/request, "
              f"out={self.max_tokens}, prompt_total="
              f"{prompt_tokens + self.tail_tokens:.0f} tok", flush=True)

    def _tail(self, idx: int) -> str:
        tail = self._tails.get(idx)
        if tail is None:
            tail = _seeded_text_for_tokens(2000 + idx, self.tail_tokens)
            self._tails[idx] = tail
        return tail

    def __call__(self) -> dict:
        model = _resolve_model_name_or_raise()
        with self._lock:
            idx = self._counter
            self._counter += 1
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.shared_prefix + "\n" + self._tail(idx)},
            ],
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

    agentic_workflow_generator = AgenticWorkflowGenerator()

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
