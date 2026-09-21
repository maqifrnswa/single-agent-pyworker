# AGENTS — operating guide for this repo

Vast.ai serverless **pyworker** for endpoint `qwen-class` (33745 / wrkgrp 45067).
The Vast base image's entrypoint (`start_server.sh`) git-CLONES this repo (`PYWORKER_REPO`)
to `/workspace/vast-pyworker`, uv-installs `requirements.txt`, and supervisor runs
`worker.py`. **Push to main IS the deploy**: every new rental/recreation picks up
current main. Existing workers are not re-provisioned by a push; a rolling update only
happens via a workergroup/template change (wipe-on-update — do not do casually).

## THE GATE (do not skip)
Before ANY push that touches `worker.py` or `workload_model.py`:

```
.venv\Scripts\python test_sdk_contract.py
```

must print `[PASS] SDK contract OK: handlers=1 init, benchmark_configs=1`.
It loads `worker.py` and executes `run()` end-to-end against the installed SDK with
only the network loop stubbed (container env dummies: `CONTAINER_ID` numeric,
`REPORT_ADDR`, `WORKER_PORT`, `VAST_TCP_PORT_<port>`, `PUBLIC_IPADDR`).
Why it exists: two consecutive main revisions died in production at startup
(`Missing EndpointHandler with BenchmarkConfig`, then a `NameError` in `run()`) —
both invisible to `py_compile`/import-smoke; engine symptom is `PyWorker exited
with status 1` looping until destroy.

## SDK invariants (verified live — do not re-litigate)
- Exactly ONE handler may carry `BenchmarkConfig` (the SDK enforces this at init).
- Raising from a benchmark generator bricks worker startup (`backend_errored`).
- `.has_benchmark` (RELATIVE path → pyworker CWD, i.e. `/workspace/vast-pyworker/.has_benchmark`)
  is written by the SDK after a successful benchmark with the float result on the first
  line; a PRESENT file short-circuits the benchmark on warm boots regardless of value.
  It lives on the persistent `/workspace` volume → survives restarts/reboots.
  Force a re-bench: delete it (do not zero it), then `supervisorctl restart pyworker`.
- `measured_perf` seen right after instance creation can be a DB-inherited stale value
  from the machine's previous workers — trust only the post-`report_benchmark` value
  (pyworker.log `average perf / measured perf` lines) and confirm the right code ran via
  the `bench: prefix=... exact_tokenizer=True` log line.

## Charge model (`workload_model.py`)
`charge = expected_output(min(max_tokens,512)) + 1.0*prompt_fresh + 0.05*prompt_cached`,
fixed weights `W_PREFILL=1.0, W_CACHED=0.05, hit_rate=0.86` (env-overridable `WORKLOAD_*`)
≈ `512 + 0.183*prompt_tokens`. This is a DELIBERATE simplification (oracle verdict:
fixed 0.86 is stable for this workload; EWMA/calibration added complexity with no
measured benefit).
**HARD INVARIANT:** `workload_model.py` must stay **byte-identical** to the copy in
`C:\Users\showard\vast_ai_controller\serverless_load\workload_model.py` — the load
harness declares client cost with the same module, so divergence breaks the
declared-cost == charge == stamp unit consistency the engine's queue math relies on.

## Benchmark design (current, `480e022`)
`BenchmarkConfig(AgenticWorkflowGenerator(), concurrency=2, runs=2)` on the chat
handler: production-shape ~61k-token shared cached prefix + ~9.94k fresh tail per
request, forced 512 decode. **max-of-runs is the intended measure** (peak capacity is
what `queue_time = cur_load/perf` wants; median would systematically under-state peak
→ over-scale). A cache-verified median warm-boot verifier was shipped (`902a2e2`) and
REVERTED (`ed102eb`) — 5-6 min added to every cold boot, unacceptable for a 1-2 GPU
fleet. Do not re-introduce it without a cost argument.

## Live validation (2026-09-21, endpoint `qwen-class`)
- Fresh stamps at this revision: 2x3090 **708.35 / 869.39 / 772.4**, A100 **1294.3**
  (spread = warmup lottery, accepted; self-corrects on later boots).
- Realized-vs-stamp 93-98% single-worker (inside the ±10% acceptance gate); the
  engine routes/autoscales on `measured_perf` (constant in traces) — the advisory
  `perf` field sloshes with reliability and is NOT the routing constant.
- First boot shows ~2-5 min of `pyworker startup paused (provisioning)` + uv
  downloads: NORMAL, not a hang.
- Stamps are unit-era-bound: numbers from pre-`480e022` code (4334 / 1198 /
  18,345-per-request charges) are incomparable — confirm era via the `bench:` log line.

## Safety
- Never print or commit keys (endpoint/workergroup JWT, `VAST_API_KEY`).
- The endpoint may serve real traffic (possibly this session's own model): keep any
  load short and user-authorized; freeze workergroup/endpoint config while a test
  run is in flight.

## Where the rest lives
Full history, run evidence (verdicts, per-worker acks), the load harness, and
trigger math: `C:\Users\showard\vast_ai_controller\serverless_load\` — `HANDOFF.md`
(closed 2026-09-21), `SCALING_TEST_PLAN.md` §11, and `../RESEARCH.md`.
