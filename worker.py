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

# Import the shared workload model (pure Python, stdlib-only).
# workload_model.py lives alongside worker.py in this repo. Worker is launched
# with the repo root as cwd, so a direct import resolves. The harness-side
# copy at vast_ai_controller/serverless_load/workload_model.py is kept in
# sync manually (they MUST stay identical -- both compute the same charge).
try:
    from workload_model import (
        charge_workload as _wm_charge_workload,
        count_tokens as _wm_count_tokens,
        set_tokenizer as _wm_set_tokenizer,
        DEFAULT_HIT_RATE as _wm_DEFAULT_HIT_RATE,
    )
    _WORKLOAD_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"workload: workload_model not available ({e}); using legacy formula", flush=True)
    _WORKLOAD_MODEL_AVAILABLE = False


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
# BENCHMARK DESIGN (steadystate): every measured request is a byte-identical
# shared prefix (a prefix-cache hit) plus a unique fresh tail (real prefill),
# and a forced full decode (ignore_eos). The shape is IDENTICAL for every
# request and every run. Payloads are sized in REAL TOKENS via count_tokens(),
# and the forced decode length equals the calculator's charged output term, so
# charge and measurement agree.
# TOKEN DENSITY NOTE: the seeded filler (random lower/digit words) tokenizes at
# ~1.5 chars/token, not the ~4 of natural text; sizing goes through
# count_tokens(), so this only affects byte length, not the token geometry.
# Slow hosts (~250 tok/s effective prefill) may still time out warming the
# prefix; they cannot serve this workload, so failing loudly beats a
# misleading score.
# Average output tokens charged per request. Measured on real opencode agentic
# traffic (text+reasoning): mean ~471, median 166, p90 ~1457. Env-overridable.
AVERAGE_OUTPUT_TOKENS = float(os.environ.get("WORKLOAD_AVG_OUTPUT_TOKENS", "512"))
# Fallback chars-per-token when the tokenizer is unavailable (~4 natural text,
# ~3.5 agentic code/JSON). Env-overridable.
WORKLOAD_CHARS_PER_TOKEN = float(os.environ.get("WORKLOAD_CHARS_PER_TOKEN", "4.0"))

# Benchmark geometry, in REAL TOKENS (production-shaped). Every measured request
# is: byte-identical shared prefix (a prefix-cache hit) + a unique fresh tail
# (real prefill), plus a forced full decode. Shape is identical for every
# request and run.
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
        # Wire up the tokenizer for the shared workload model.
        if _TOKENIZER is not None and _WORKLOAD_MODEL_AVAILABLE:
            def _tok_fn(text):
                return len(_TOKENIZER.encode(text, add_special_tokens=False).ids)
            _wm_set_tokenizer(_tok_fn)
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


def chat_workload(data) -> float:
    """Charged workload using the three-component formula with fixed hit_rate.

    Formula:
      work = expected_output_tokens
           + W_PREFILL * prompt_tokens * (1 - hit_rate)
           + W_CACHED  * prompt_tokens * hit_rate

    hit_rate is the fixed DEFAULT_HIT_RATE (0.86 measured steady-state cache
    hit rate; no EWMA tracking in this redesign). The charge is computed by the
    shared workload_model so the SDK charge and the background benchmark agree.
    """
    try:
        prompt_tokens = _wm_count_tokens(_prompt_text(data))
        return _wm_charge_workload(data, prompt_tokens=prompt_tokens, hit_rate=_wm_DEFAULT_HIT_RATE)
    except Exception:
        # Fallback to SDK's default if workload_model is unavailable or charge fails.
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
        """Generate a benchmark payload (shared prefix + unique fresh tail).

        Cache verification is NOT enforced here (lib-1: raising inside the SDK's
        benchmark generator bricks the worker with backend_errored). The
        background benchmark loop verifies cache on the responses instead.
        """
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

    # Relative path resolves against the server url+port; a full URL is used as-is.
    healthcheck_url = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")

    # Production-shaped benchmark payload generator (shared cached prefix +
    # unique fresh tail). Constructed ONCE; referenced by the chat handler's
    # BenchmarkConfig below. (NameError at run() time if this line is missing
    # -- test_sdk_contract.py catches that; py_compile cannot.)
    agentic_workflow_generator = AgenticWorkflowGenerator()

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
                # BenchmarkConfig is REQUIRED by the SDK (server/worker.py:380
                # raises "Missing EndpointHandler with BenchmarkConfig" if no
                # handler has one). We attach ours to chat so the workload_calculator
                # (3-component formula) and the benchmark charge are in matching
                # units -- otherwise queue_time = cur_load / perf is dimensionally
                # wrong. Same AgenticWorkflowGenerator as before: production-shaped
                # (cached prefix + fresh tail), token-sized, ignore_eos for
                # forced full decode. runs=2 per the assessment: measured runs
                # are all warm, so runs=2 is stable. The SDK reports max-of-runs.
                benchmark_config=BenchmarkConfig(
                    generator=agentic_workflow_generator,
                    concurrency=2,
                    runs=2,
                ),
            ),
        ],
        log_action_config=LogActionConfig(
            on_load=_env_lines("MODEL_LOAD_LOG_MSG", defaults.load_log_msgs),
            on_error=_env_lines("MODEL_ERROR_LOG_MSGS", defaults.error_log_msgs),
            on_info=_env_lines("MODEL_INFO_LOG_MSGS", defaults.info_log_msgs),
        ),
    )
    # No cache-verification verifier: cache is deterministic for byte-
    # identical prefix (AgenticWorkflowGenerator seeds this); the observed
    # 431/689/897 spread across restarts was SDK warmup reliability, not
    # cache behavior. A 5x2 verifier would add ~5-6 min per boot -- too
    # expensive on a 1-2 GPU fleet. SDK's max-of-2 is directionally correct;
    # the ~2x worst-case variance is acceptable for this fleet size.
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
