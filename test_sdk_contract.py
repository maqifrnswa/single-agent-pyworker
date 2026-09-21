"""LOCAL SDK CONTRACT TEST: build the same WorkerConfig structure the pyworker
builds and run it through the installed vastai SDK's validation -- WITHOUT
starting the network loop. This is the test that should have existed before
every push: it deterministically catches the 'Missing EndpointHandler with
BenchmarkConfig' class of crash (and the inverse, >1 BenchmarkConfig).

Usage: python test_sdk_contract.py [path/to/worker.py]

Strategy: importlib-load worker.py in a patched environment where
Worker/WorkerConfig are real SDK objects but Worker.run is replaced with a
no-op; then call worker.run(defaults) with fake env. If __init__ validation
raises, we see it. If run() is stubbed, no network happens.
"""
import os
import sys
import types

WORKER_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\showard\single-agent-pyworker\worker.py"

# 1) Make the pyworker repo importable (workload_model.py sits next to worker.py)
repo_dir = os.path.dirname(os.path.abspath(WORKER_PATH))
sys.path.insert(0, repo_dir)

# 2) Env the module needs at import time (tokenizer may be missing -> fallback)
os.environ.setdefault("MODEL_NAME", "/models/fake-model")
# Container-only env that the real Vast entrypoint sets (captured from a live
# box's /workspace/debug.log + SDK source serverless/server/lib/metrics.py):
# Worker init reads CONTAINER_ID (int), REPORT_ADDR, WORKER_PORT,
# VAST_TCP_PORT_<WORKER_PORT>, PUBLIC_IPADDR. Dummies are fine -- Worker.run
# is stubbed below so nothing connects to REPORT_ADDR.
os.environ.setdefault("CONTAINER_ID", "99999999")   # SDK int()s this
os.environ.setdefault("REPORT_ADDR", "https://run.vast.ai.invalid-test")
os.environ.setdefault("WORKER_PORT", "3000")
os.environ.setdefault("VAST_TCP_PORT_3000", "3000")
os.environ.setdefault("PUBLIC_IPADDR", "127.0.0.1")
os.environ.setdefault("BACKEND", "vllm")

import vastai
import vastai.serverless.server.worker as sdkw
print(f"vastai SDK file: {sdkw.__file__}")

# 3) Import the pyworker module
import importlib.util
spec = importlib.util.spec_from_file_location("pyworker", WORKER_PATH)
pw = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(pw)
except Exception as e:
    print(f"[FAIL] module import: {type(e).__name__}: {e}")
    sys.exit(2)
print("[ok] pyworker module imports")

# 4) Stub Worker.run so nothing connects; keep real __init__ validation.
calls = {"init": 0, "run": 0, "benchmark_handler_ok": 0}
RealWorker = pw.Worker

class SpyWorker(RealWorker):
    def __init__(self, config):
        calls["init"] += 1
        super().__init__(config)          # THIS is where 'Missing EndpointHandler' raised
        # after successful init, prove the benchmark handler resolved:
        try:
            cfg = config
            for h in cfg.handlers:
                if getattr(h, "benchmark_config", None):
                    calls["benchmark_handler_ok"] += 1
        except Exception:
            pass

    def run(self):
        calls["run"] += 1
        print("[stub] Worker.run() called -- validation PASSED, no network")

pw.Worker = SpyWorker

# 5) Build the config exactly as production does
defaults = pw.EngineDefaults(
    name="vllm",
    model_log_file="/var/log/portal/vllm.log",
    load_log_msgs=["Application startup complete.", "engines connected"],
    error_log_msgs=["INFO exited: vllm", "EngineCore failed to start"],
)
try:
    pw.run(defaults)
except Exception as e:
    print(f"[FAIL] Worker config validation raised: {type(e).__name__}: {e}")
    sys.exit(3)

if calls["init"] == 1 and calls["benchmark_handler_ok"] == 1:
    print(f"[PASS] SDK contract OK: handlers={calls['init']} init, "
          f"benchmark_configs={calls['benchmark_handler_ok']} (need exactly 1)")
    sys.exit(0)
else:
    print(f"[FAIL] unexpected contract state: {calls}")
    sys.exit(4)