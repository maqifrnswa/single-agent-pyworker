"""Repro for depth-10 400s: build the exact benchmark payload and POST it.

Usage (no network):  set MODEL_NAME and run; prints payload stats only.
Usage (full repro):  also set ENDPOINT_URL (and VAST_API_KEY if needed);
                     posts one depth-10 payload and prints status + body.
"""
import json
import os
import urllib.error
import urllib.request

os.environ.setdefault("MODEL_NAME", "/models/qwen3.8-27b-awq-int4")

from worker import AgenticWorkflowGenerator

depth = int(os.environ.get("DEPTH", "10"))
gen = AgenticWorkflowGenerator(depths=(depth,), max_tokens=512)
payload = gen()

n_chars = sum(len(m.get("content", "")) for m in payload["messages"])
print("num_messages:", len(payload["messages"]))
print("total_chars:", n_chars, "est_tokens:", round(n_chars / 4))
print("max_tokens:", payload["max_tokens"], "ignore_eos:", payload.get("ignore_eos"))
print("keys:", sorted(payload.keys()))

with open("depth10_payload.json", "w", encoding="utf-8") as f:
    json.dump(payload, f)
print("wrote depth10_payload.json")

endpoint = os.environ.get("ENDPOINT_URL")
if not endpoint:
    print("ENDPOINT_URL unset: stats only, no POST.")
    raise SystemExit(0)

body = json.dumps(payload).encode("utf-8")
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
