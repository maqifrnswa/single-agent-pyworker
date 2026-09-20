"""Payload-stats tool for the token-sized benchmark (formerly the depth-10 repro).

Builds the exact benchmark payload with worker.AgenticWorkflowGenerator and
prints its shape: shared cached prefix + unique fresh tail, token-sized.

Usage (no network):  set MODEL_NAME and run; prints payload stats only.
Usage (full repro):  also set ENDPOINT_URL (and VAST_API_KEY if needed);
                     posts one payload and prints status + body.
"""
import json
import os
import urllib.error
import urllib.request

os.environ.setdefault("MODEL_NAME", "/models/qwen3.8-27b-awq-int4")

from worker import AgenticWorkflowGenerator, _prompt_text, count_tokens

gen = AgenticWorkflowGenerator()
p1 = gen()
p2 = gen()

prefix = gen.shared_prefix
u1 = p1["messages"][-1]["content"]
u2 = p2["messages"][-1]["content"]
tail1 = u1[len(prefix) + 1:] if u1.startswith(prefix + "\n") else None
tail2 = u2[len(prefix) + 1:] if u2.startswith(prefix + "\n") else None

t1 = count_tokens(_prompt_text(p1))
t2 = count_tokens(_prompt_text(p2))

print("num_messages:", len(p1["messages"]))
print("prefix_sha256:", gen.prefix_hash)
print("prefix_tokens:", int(count_tokens(prefix)))
print("tail_tokens:", int(count_tokens(tail1)) if tail1 is not None else None)
print("prompt_tokens_p1:", t1)
print("prompt_tokens_p2:", t2)
# Token counts are integers; the estimator returns a float, so equality is on
# whole tokens (per-seed sizing can leave a sub-token/1-char residual).
print("prompt_tokens_equal:", int(t1) == int(t2))
print("tails_distinct:", tail1 is not None and tail2 is not None and tail1 != tail2)
print("shared_prefix_identical:",
      u1.startswith(prefix) and u2.startswith(prefix) and prefix in u1)
print("max_tokens:", p1["max_tokens"], "ignore_eos:", p1.get("ignore_eos"))
print("keys:", sorted(p1.keys()))

with open("bench_payload.json", "w", encoding="utf-8") as f:
    json.dump(p1, f)
print("wrote bench_payload.json")

endpoint = os.environ.get("ENDPOINT_URL")
if not endpoint:
    print("ENDPOINT_URL unset: stats only, no POST.")
    raise SystemExit(0)

body = json.dumps(p1).encode("utf-8")
print("payload_bytes:", len(body))
headers = {"Content-Type": "application/json"}
api_key = os.environ.get("VAST_API_KEY")
if api_key:
    headers["Authorization"] = "Bearer " + api_key
req = urllib.request.Request(
    os.environ.get("FULL_URL", endpoint.rstrip("/") + "/v1/chat/completions"),
    data=body,
    headers=headers,
)
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        print("HTTP", r.status)
        print(r.read()[:500])
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read()[:2000].decode("utf-8", "replace"))
