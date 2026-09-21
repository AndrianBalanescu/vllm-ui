#!/usr/bin/env python3
import http.server
import socketserver
import urllib.request
import urllib.error
import argparse
import json
import subprocess
import os
import sys
import re
import time
import uuid
import threading
from collections import deque
from pathlib import Path

# CLI & Environment Defaults
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"
DEFAULT_VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "")
DEFAULT_DASHBOARD_TOKEN = os.environ.get("VLLM_DASHBOARD_TOKEN", "")
MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MB
MAX_CONCURRENT_REQUESTS = 32

STATIC_DIR = Path(__file__).parent / "static" if "__file__" in globals() else None

class Config:
    port = DEFAULT_PORT
    host = DEFAULT_HOST
    vllm_url = DEFAULT_VLLM_URL
    api_key = DEFAULT_API_KEY
    dashboard_token = DEFAULT_DASHBOARD_TOKEN

# Concurrency guard: limits concurrent proxy requests to vLLM
_active_proxy_count = 0
_proxy_lock = threading.Lock()

class RequestTracker:
    def __init__(self, max_len=500):
        self.lock = threading.RLock()
        self.requests = deque(maxlen=max_len)
        self.active_requests = {}
        self.engine_slots = {}
        self.last_sync_time = time.time()

    def start_request(self, model="default", prompt=""):
        req_id = str(uuid.uuid4())
        now = time.time()
        record = {
            "id": req_id,
            "model": model,
            "prompt_preview": (prompt[:60] + "...") if len(prompt) > 60 else prompt,
            "start_time": now,
            "end_time": None,
            "duration_ms": 0,
            "status": "active",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tps": 0.0
        }
        with self.lock:
            self.active_requests[req_id] = record
        return req_id

    def finish_request(self, req_id, status_code=200, prompt_tokens=0, completion_tokens=0, model=None):
        now = time.time()
        with self.lock:
            record = self.active_requests.pop(req_id, None)
            if not record:
                return
            duration_s = max(now - record["start_time"], 0.05)
            duration_ms = int(duration_s * 1000)
            record["end_time"] = now
            record["duration_ms"] = duration_ms
            record["status"] = status_code
            record["prompt_tokens"] = prompt_tokens
            record["completion_tokens"] = completion_tokens
            record["tps"] = round(completion_tokens / duration_s, 1) if completion_tokens > 0 else 0.0
            if model:
                record["model"] = model
            self.requests.append(record)

    def update_slot_task(self, slot_id, model="default", prompt_tokens=0, cached_tokens=0, processed=0):
        now = time.time()
        with self.lock:
            if slot_id not in self.engine_slots:
                preview = f"Engine Request ({prompt_tokens:,} tok"
                if cached_tokens > 0:
                    preview += f", {cached_tokens:,} cached"
                preview += ")"
                self.engine_slots[slot_id] = {
                    "id": slot_id,
                    "model": model,
                    "prompt_preview": preview,
                    "start_time": now,
                    "end_time": None,
                    "duration_ms": 0,
                    "status": "active",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": processed,
                    "tps": 0.0
                }
            else:
                rec = self.engine_slots[slot_id]
                rec["prompt_tokens"] = prompt_tokens
                rec["completion_tokens"] = processed
                rec["duration_ms"] = int((now - rec["start_time"]) * 1000)

    def sync_engine_slots(self, running_count, default_model="vLLM"):
        now = time.time()
        with self.lock:
            active_manual_count = len(self.active_requests)
            target_engine_count = max(0, int(running_count) - active_manual_count)

            while len(self.engine_slots) < target_engine_count:
                slot_id = f"slot-{uuid.uuid4().hex[:8]}"
                self.engine_slots[slot_id] = {
                    "id": slot_id,
                    "model": default_model,
                    "prompt_preview": "In-Flight Engine Execution",
                    "start_time": now,
                    "end_time": None,
                    "duration_ms": 0,
                    "status": "active",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tps": 0.0
                }

            while len(self.engine_slots) > target_engine_count:
                oldest_key = min(self.engine_slots.keys(), key=lambda k: self.engine_slots[k]["start_time"])
                record = self.engine_slots.pop(oldest_key)
                duration_s = max(now - record["start_time"], 0.05)
                record["end_time"] = now
                record["duration_ms"] = int(duration_s * 1000)
                record["status"] = 200
                record["completion_tokens"] = 0
                record["tps"] = 0.0
                self.requests.append(record)

            for slot_id, rec in list(self.engine_slots.items()):
                rec["duration_ms"] = int((now - rec["start_time"]) * 1000)

    def get_timeline(self):
        now = time.time()
        with self.lock:
            completed_list = list(self.requests)
            active_list = list(self.active_requests.values()) + list(self.engine_slots.values())

            for r in active_list:
                if r["end_time"] is None:
                    r["duration_ms"] = int((now - r["start_time"]) * 1000)

            all_items = completed_list + active_list
            all_items.sort(key=lambda x: x["start_time"])
            return all_items[-150:]

tracker = RequestTracker()

def get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "mem_used_mb": float(parts[1]),
            "mem_total_mb": float(parts[2]),
            "util_percent": float(parts[3]),
            "temp_c": float(parts[4]),
            "power_draw_w": float(parts[5]) if len(parts) > 5 else 0,
            "power_limit_w": float(parts[6]) if len(parts) > 6 else 0
        }
    except Exception:
        return {"name": "GPU Host", "mem_used_mb": 0, "mem_total_mb": 0, "util_percent": 0, "temp_c": 0, "power_draw_w": 0, "power_limit_w": 0}

def parse_prometheus_metrics(text):
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([0-9.eE+-]+)", line)
        if m:
            name, labels_str, val_str = m.group(1), m.group(2), m.group(3)
            try:
                val = float(val_str)
            except ValueError:
                val = 0.0

            if labels_str:
                if name not in metrics:
                    metrics[name] = []
                labels = {}
                for kv in re.finditer(r'([a-zA-Z0-9_]+)="([^"]*)"', labels_str):
                    labels[kv.group(1)] = kv.group(2)
                metrics[name].append({"labels": labels, "value": val})
            else:
                metrics[name] = val
    return metrics

def get_recent_vllm_logs():
    try:
        out = subprocess.check_output(
            ["sudo", "journalctl", "-u", "swift-qwen", "-u", "vllm", "-u", "vllm-qwen", "-n", "80", "--no-pager"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        lines = []
        for l in out.splitlines():
            if any(k in l for k in ["prompt processing", "print_timing", "eval time", "HTTP", "ERROR", "POST /v1", "Engine", "INFO", "release"]):
                lines.append(l.strip())
        return lines[-40:]
    except Exception:
        return []

_models_cache = {"t": 0.0, "val": (["default"], None)}

def get_models_info():
    global _models_cache
    now = time.time()
    if now - _models_cache["t"] < 30:
        return _models_cache["val"]
    try:
        url = f"{Config.vllm_url.rstrip('/')}/v1/models"
        headers = {}
        if Config.api_key:
            headers["Authorization"] = f"Bearer {Config.api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_data = [m for m in data.get("data", []) if not str(m.get("id", "")).startswith("modelperm-")]
            models = [m["id"] for m in models_data]
            max_model_len = None
            if models_data and "max_model_len" in models_data[0]:
                max_model_len = models_data[0]["max_model_len"]
            val = (models if models else ["default"], max_model_len)
            _models_cache = {"t": now, "val": val}
            return val
    except Exception:
        return _models_cache["val"]

def get_models_list():
    models, _ = get_models_info()
    return models

_last_sample_time = time.time()
_last_gen_tokens = 0
_last_prompt_tokens = 0
_peak_prompt_tps = 0.0
_peak_gen_tps = 0.0

_cache_lock = threading.Lock()
_stats_cache = {"timestamp": 0.0, "data": None}

def _build_stats():
    global _last_sample_time, _last_gen_tokens, _last_prompt_tokens, _peak_prompt_tps, _peak_gen_tps

    gpu = get_gpu_stats()
    metrics_raw = ""
    try:
        url = f"{Config.vllm_url.rstrip('/')}/metrics"
        headers = {}
        if Config.api_key:
            headers["Authorization"] = f"Bearer {Config.api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            metrics_raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        metrics_raw = ""

    parsed = parse_prometheus_metrics(metrics_raw) if metrics_raw else {}

    def get_single(key, default=None):
        val = parsed.get(key, None)
        if val is None:
            return default
        if isinstance(val, list) and len(val) > 0:
            return val[0]["value"]
        return float(val) if isinstance(val, (int, float)) else default

    running = get_single("vllm:num_requests_running")
    if running is None:
        running = get_single("llamacpp:requests_processing", 0.0)

    waiting = get_single("vllm:num_requests_waiting")
    if waiting is None:
        waiting = get_single("llamacpp:requests_deferred", 0.0)

    kv_cache_factor = get_single("vllm:kv_cache_usage_perc")
    if kv_cache_factor is None:
        kv_cache_factor = get_single("vllm:gpu_cache_usage_factor")
    if kv_cache_factor is None:
        kv_cache_factor = get_single("llamacpp:kv_cache_usage_ratio", 0.0)
    kv_cache_usage = kv_cache_factor * 100.0

    prompt_throughput = get_single("vllm:avg_prompt_throughput_tok_per_s", 0.0)
    gen_throughput = get_single("vllm:avg_generation_throughput_tok_per_s", 0.0)

    total_prompt_tokens = get_single("vllm:prompt_tokens_total")
    if total_prompt_tokens is None:
        total_prompt_tokens = get_single("llamacpp:prompt_tokens_total", 0.0)

    total_gen_tokens = get_single("vllm:generation_tokens_total")
    if total_gen_tokens is None:
        total_gen_tokens = get_single("llamacpp:tokens_predicted_total", 0.0)

    total_requests = get_single("vllm:time_to_first_token_seconds_count")
    if total_requests is None:
        total_requests = get_single("llamacpp:requests_completed_total")
    if total_requests is None or total_requests == 0.0:
        tracker_t = tracker.get_totals()
        total_requests = tracker_t["total_requests"]
        if total_requests == 0 and (total_prompt_tokens > 0 or total_gen_tokens > 0):
            total_requests = max(1.0, float(len(tracker.requests) + len(tracker.engine_slots)))

    now = time.time()
    dt = max(now - _last_sample_time, 0.5)

    if _last_sample_time > 0 and _last_gen_tokens > 0 and total_gen_tokens >= _last_gen_tokens:
        delta_gen_tps = (total_gen_tokens - _last_gen_tokens) / dt
        if delta_gen_tps > 0:
            gen_throughput = delta_gen_tps

    if _last_sample_time > 0 and _last_prompt_tokens > 0 and total_prompt_tokens >= _last_prompt_tokens:
        delta_prompt_tps = (total_prompt_tokens - _last_prompt_tokens) / dt
        if delta_prompt_tps > 0:
            prompt_throughput = delta_prompt_tps

    _last_sample_time = now
    _last_gen_tokens = total_gen_tokens
    _last_prompt_tokens = total_prompt_tokens

    ttft_sum = get_single("vllm:time_to_first_token_seconds_sum")
    ttft_count = get_single("vllm:time_to_first_token_seconds_count")
    avg_ttft_ms = round((ttft_sum / ttft_count * 1000.0), 1) if ttft_count and ttft_sum else None

    e2e_sum = get_single("vllm:e2e_request_latency_seconds_sum")
    e2e_count = get_single("vllm:e2e_request_latency_seconds_count")
    avg_latency_ms = round((e2e_sum / e2e_count * 1000.0), 1) if e2e_count and e2e_sum else None

    accepted_drafts = get_single("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    total_drafts = get_single("vllm:spec_decode_num_draft_tokens_total", 0.0)
    spec_acceptance_pct = round((accepted_drafts / total_drafts * 100.0), 1) if total_drafts > 0 else None

    cached_tokens = (
        get_single("vllm:prompt_tokens_cached_total")
        or get_single("vllm:prefix_cache_hits_total")
        or get_single("vllm:num_cached_tokens_total")
    )
    if cached_tokens is None:
        cached_tokens = get_single("llamacpp:prompt_tokens_cached_total", 0.0)

    prefix_queries = get_single("vllm:prefix_cache_queries_total", 0.0)
    if prefix_queries > 0:
        prefix_hit_pct = round((cached_tokens / prefix_queries * 100.0), 1)
    elif (cached_tokens + total_prompt_tokens) > 0:
        prefix_hit_pct = round((cached_tokens / (cached_tokens + total_prompt_tokens) * 100.0), 1)
    else:
        prefix_hit_pct = 0.0

    logs = get_recent_vllm_logs()
    instant_prompt_tps = prompt_throughput
    instant_gen_tps = gen_throughput
    instant_kv_pct = kv_cache_usage
    instant_prefix_pct = prefix_hit_pct

    for l in reversed(logs):
        if "Engine 000:" in l or "Engine" in l:
            m_p = re.search(r"Avg prompt throughput:\s*([\d.]+)", l)
            m_g = re.search(r"Avg generation throughput:\s*([\d.]+)", l)
            m_kv = re.search(r"GPU KV cache usage:\s*([\d.]+)%", l)
            m_pre = re.search(r"Prefix cache hit rate:\s*([\d.]+)%", l)
            if m_p and float(m_p.group(1)) > 0 and instant_prompt_tps == 0:
                instant_prompt_tps = float(m_p.group(1))
            if m_g and float(m_g.group(1)) > 0 and instant_gen_tps == 0:
                instant_gen_tps = float(m_g.group(1))
            if m_kv: instant_kv_pct = float(m_kv.group(1))
            if m_pre: instant_prefix_pct = float(m_pre.group(1))
            break
        # Also parse llama.cpp logs
        m_lp = re.search(r"prompt processing,.*?[/\s]([\d.]+)\s*tokens per second", l)
        if not m_lp:
            m_lp = re.search(r"prompt eval time =.*?([\d.]+)\s*tokens per second", l)
        if m_lp and float(m_lp.group(1)) > 0 and instant_prompt_tps == 0:
            instant_prompt_tps = float(m_lp.group(1))

        m_lg = re.search(r"tg = ([\d.]+) t/s", l)
        if not m_lg:
            m_lg = re.search(r"eval time =.*?([\d.]+)\s*tokens per second", l)
        if m_lg and float(m_lg.group(1)) > 0 and instant_gen_tps == 0:
            instant_gen_tps = float(m_lg.group(1))

    models, max_model_len = get_models_info()
    active_model = models[0] if models else "vLLM"

    # Poll llama.cpp /slots to show active requests and token counts in UI table
    slots_active = 0
    try:
        url_slots = f"{Config.vllm_url.rstrip('/')}/slots"
        req_slots = urllib.request.Request(url_slots)
        with urllib.request.urlopen(req_slots, timeout=1) as resp:
            slots_list = json.loads(resp.read().decode())
            for s in slots_list:
                if s.get("is_processing"):
                    slots_active += 1
                    t_id = f"task-{s.get('id_task', s.get('id', 0))}"
                    p_tok = s.get("n_prompt_tokens", 0)
                    c_tok = s.get("n_prompt_tokens_cache", 0)
                    proc_tok = s.get("n_prompt_tokens_processed", 0)
                    tracker.update_slot_task(t_id, model=active_model, prompt_tokens=p_tok, cached_tokens=c_tok, processed=proc_tok)
    except Exception:
        pass

    if slots_active > 0:
        running = max(running, float(slots_active))

    if instant_prompt_tps > _peak_prompt_tps:
        _peak_prompt_tps = instant_prompt_tps
    if instant_gen_tps > _peak_gen_tps:
        _peak_gen_tps = instant_gen_tps

    tracker.sync_engine_slots(running, default_model=active_model)
    timeline = tracker.get_timeline()

    return {
        "timestamp": time.time(),
        "gpu": gpu,
        "totals": {
            "total_requests": int(total_requests),
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_gen_tokens": int(total_gen_tokens),
            "total_tokens": int(total_prompt_tokens + total_gen_tokens),
            "avg_ttft_ms": avg_ttft_ms,
            "avg_latency_ms": avg_latency_ms,
            "accepted_draft_tokens": int(accepted_drafts),
            "total_draft_tokens": int(total_drafts)
        },
        "engine": {
            "status": "online" if metrics_raw else "offline",
            "running_requests": int(running),
            "waiting_requests": int(waiting),
            "gen_throughput_tps": round(instant_gen_tps, 1),
            "prompt_throughput_tps": round(instant_prompt_tps, 1),
            "peak_gen_tps": round(_peak_gen_tps, 1),
            "peak_prompt_tps": round(_peak_prompt_tps, 1),
            "kv_cache_usage_pct": round(instant_kv_pct, 1),
            "prefix_cache_hit_pct": round(instant_prefix_pct, 1),
            "spec_acceptance_pct": spec_acceptance_pct,
            "models": models,
            "active_model": active_model,
            "max_model_len": max_model_len,
            "vllm_url": Config.vllm_url
        },
        "timeline": timeline,
        "recent_logs": logs
    }

def get_stats():
    with _cache_lock:
        data = _stats_cache["data"]
        if data is not None:
            return data
    return _build_stats()

def _run_stats_sampler():
    while True:
        try:
            data = _build_stats()
            with _cache_lock:
                _stats_cache["data"] = data
        except Exception:
            pass
        time.sleep(1.0)

def start_samplers():
    t = threading.Thread(target=_run_stats_sampler, name="stats-sampler", daemon=True)
    t.start()
EMBEDDED_INDEX_HTML = "<!DOCTYPE html>\n<html lang=\"en\" class=\"dark\">\n<head>\n  <meta charset=\"UTF-8\">\n  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n  <title>vLLM \u2022 Live Telemetry & Control Dashboard</title>\n  <!-- Tailwind CSS (local build, zero runtime deps) -->\n  <link rel=\"stylesheet\" href=\"/tailwind.css\">\n  <!-- Vue 3 Global -->\n  <script src=\"https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.prod.js\"></script>\n  <!-- Chart.js -->\n  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n  <!-- Lucide Icons -->\n  <script src=\"https://unpkg.com/lucide@latest\"></script>\n  <!-- Google Fonts -->\n  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n  <link href=\"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap\" rel=\"stylesheet\">\n\n\n  <style>\n    body {\n      background-color: #06090f;\n      color: #f1f5f9;\n      font-family: 'Plus Jakarta Sans', sans-serif;\n    }\n    .glass-card {\n      background: rgba(15, 23, 42, 0.75);\n      backdrop-filter: blur(12px);\n      border: 1px solid rgba(255, 255, 255, 0.07);\n    }\n    .glass-card:hover {\n      border-color: rgba(16, 185, 129, 0.35);\n    }\n    /* Custom scrollbar */\n    ::-webkit-scrollbar {\n      width: 6px;\n      height: 6px;\n    }\n    ::-webkit-scrollbar-track {\n      background: #0a0f1d;\n    }\n    ::-webkit-scrollbar-thumb {\n      background: #1e293b;\n      border-radius: 4px;\n    }\n    ::-webkit-scrollbar-thumb:hover {\n      background: #10b981;\n    }\n  </style>\n</head>\n<body class=\"min-h-screen antialiased selection:bg-emerald-500 selection:text-black\">\n  <div id=\"app\" class=\"max-w-[1700px] mx-auto p-4 sm:p-5 lg:p-6 space-y-4\">\n    <!-- Stale Data Warning -->\n    <div v-if=\"isStatsStale()\" class=\"mb-2 px-4 py-2 rounded-lg bg-amber-950/80 border border-amber-800/50 text-amber-300 text-xs font-mono flex items-center gap-2\">\n      <span class=\"w-2 h-2 rounded-full bg-amber-400 animate-pulse\"></span>\n      Data stale (&gt;5s). vLLM backend may be unreachable.\n    </div>\n    \n    <!-- Top Header & Appliance Status Bar -->\n    <header class=\"flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80\">\n      <div class=\"flex items-center gap-3\">\n        <div class=\"w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-950/60\">\n          <i data-lucide=\"zap\" class=\"w-5 h-5 text-black font-bold\"></i>\n        </div>\n        <div>\n          <div class=\"flex items-center gap-2.5\">\n            <h1 class=\"text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent\">\n              vLLM Telemetry & Control Center\n            </h1>\n            <span class=\"inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800/50\">\n              <span class=\"w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse\"></span>\n              {{ stats.gpu?.name || 'GPU Engine' }}\n            </span>\n          </div>\n          <div class=\"flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400 font-mono mt-0.5\">\n            <span>Model: <strong class=\"text-slate-200\">{{ stats.engine?.active_model || '--' }}</strong></span>\n            <span class=\"text-slate-600\" v-if=\"stats.engine?.max_model_len\">\u2022</span>\n            <span v-if=\"stats.engine?.max_model_len\">Ctx: {{ formatNumber(stats.engine.max_model_len) }}</span>\n            <span class=\"text-slate-600\" v-if=\"stats.engine?.spec_acceptance_pct !== null && stats.engine?.spec_acceptance_pct !== undefined\">\u2022</span>\n            <span v-if=\"stats.engine?.spec_acceptance_pct !== null && stats.engine?.spec_acceptance_pct !== undefined\">MTP / Spec: {{ stats.engine.spec_acceptance_pct }}%</span>\n          </div>\n        </div>\n      </div>\n\n      <!-- Services Status Pills & Global Controls -->\n      <div class=\"flex flex-wrap items-center gap-2\">\n        <div class=\"flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-800 text-xs font-mono\">\n          <span class=\"w-2 h-2 rounded-full\" :class=\"stats.engine?.status === 'online' ? 'bg-emerald-400' : 'bg-rose-500'\"></span>\n          <span class=\"text-slate-300\">vLLM Engine</span>\n          <span class=\"text-slate-500 text-[10px]\">:8000</span>\n        </div>\n\n        <button @click=\"showCurlModal = true\" class=\"flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-800/90 hover:bg-slate-700 text-emerald-400 hover:text-emerald-300 border border-slate-700 text-xs font-mono transition\">\n          <i data-lucide=\"terminal\" class=\"w-3.5 h-3.5\"></i>\n          <span>cURL / API</span>\n        </button>\n\n        <button @click=\"fetchStats(true)\" class=\"p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white transition border border-slate-700\" title=\"Refresh\">\n          <i data-lucide=\"rotate-cw\" class=\"w-3.5 h-3.5\" :class=\"{'animate-spin': isRefreshing}\"></i>\n        </button>\n      </div>\n    </header>\n\n    <!-- LIFETIME TELEMETRY & TOTALS STATS STRIP (PROMINENT TOP BANNER) -->\n    <div class=\"glass-card rounded-2xl px-4 sm:px-5 py-3 sm:py-3.5 font-mono border border-slate-700/60 bg-slate-900/90 shadow-xl shadow-black/50\">\n      <div class=\"flex flex-wrap items-center gap-x-6 gap-y-2 sm:gap-x-10\">\n        <div class=\"flex items-center gap-2 sm:gap-2.5\">\n          <span class=\"text-slate-400 uppercase tracking-wider text-xs font-bold\">Requests:</span>\n          <span class=\"text-white text-base font-extrabold bg-slate-800/90 px-2.5 sm:px-3 py-0.5 rounded-lg border border-slate-700 shadow-sm tabular-nums min-w-[64px] text-center\">{{ formatNumber(stats.totals?.total_requests || 0) }}</span>\n        </div>\n\n        <div class=\"flex items-center gap-2 sm:gap-2.5\">\n          <span class=\"text-slate-400 uppercase tracking-wider text-xs font-bold\">Tokens:</span>\n          <span class=\"text-emerald-400 text-base font-extrabold tracking-tight tabular-nums min-w-[56px] text-right\">{{ formatCompactNumber(stats.totals?.total_tokens || 0) }}</span>\n          <span class=\"text-slate-500 text-[11px] font-medium tabular-nums hidden sm:inline\">({{ formatCompactNumber(stats.totals?.total_prompt_tokens || 0) }}p / {{ formatCompactNumber(stats.totals?.total_gen_tokens || 0) }}g)</span>\n        </div>\n\n        <div class=\"flex items-center gap-2 sm:gap-2.5\">\n          <span class=\"text-slate-400 uppercase tracking-wider text-xs font-bold\">TTFT:</span>\n          <span class=\"text-teal-300 text-base font-bold tabular-nums min-w-[64px] text-right\">{{ stats.totals?.avg_ttft_ms ? formatMs(stats.totals.avg_ttft_ms) : '--' }}</span>\n        </div>\n\n        <div class=\"flex items-center gap-2 sm:gap-2.5\">\n          <span class=\"text-slate-400 uppercase tracking-wider text-xs font-bold\">Latency:</span>\n          <span class=\"text-blue-300 text-base font-bold tabular-nums min-w-[64px] text-right\">{{ stats.totals?.avg_latency_ms ? formatMs(stats.totals.avg_latency_ms) : '--' }}</span>\n        </div>\n\n        <div class=\"flex items-center gap-2 sm:gap-2.5\" v-if=\"stats.totals?.accepted_draft_tokens\">\n          <span class=\"text-slate-400 uppercase tracking-wider text-xs font-bold hidden md:inline\">MTP Draft:</span>\n          <span class=\"text-emerald-400 text-base font-bold tabular-nums min-w-[56px] text-right\">{{ formatCompactNumber(stats.totals.accepted_draft_tokens) }}</span>\n        </div>\n      </div>\n\n      <div class=\"flex items-center gap-2 text-[11px] text-slate-500 font-semibold mt-2.5 pt-2.5 border-t border-slate-800/60\">\n        <span class=\"inline-block w-2 h-2 rounded-full bg-emerald-400 animate-pulse\"></span>\n        <span>Zero-overhead engine metrics</span>\n      </div>\n    </div>\n\n    <!-- Compact Key Stats Cards Grid -->\n    <div class=\"grid grid-cols-2 lg:grid-cols-4 gap-3\">\n      \n      <!-- Card 1: Throughput tok/s -->\n      <div class=\"glass-card rounded-xl p-3.5 transition-all duration-200\">\n        <div class=\"flex items-center justify-between text-slate-400 mb-1\">\n          <span class=\"text-[11px] font-bold uppercase tracking-wider text-slate-400\">Generation Speed</span>\n          <i data-lucide=\"zap\" class=\"w-3.5 h-3.5 text-emerald-400\"></i>\n        </div>\n        <div class=\"flex items-baseline gap-1.5\">\n          <span class=\"text-2xl font-extrabold font-mono text-white tabular-nums min-w-[84px]\">{{ formatTps(stats.engine?.gen_throughput_tps || 0) }}</span>\n          <span class=\"text-[11px] font-mono text-emerald-400 font-bold\">tok/s</span>\n        </div>\n        <div class=\"mt-2 flex items-center justify-between text-[11px] text-slate-400 font-mono pt-2 border-t border-slate-800/80\">\n          <span>Prefill Speed:</span>\n          <span class=\"text-slate-200 font-semibold tabular-nums\">{{ formatTps(stats.engine?.prompt_throughput_tps || 0) }} tok/s</span>\n        </div>\n      </div>\n\n      <!-- Card 2: KV Cache Memory Pool -->\n      <div class=\"glass-card rounded-xl p-3.5 transition-all duration-200\">\n        <div class=\"flex items-center justify-between text-slate-400 mb-1\">\n          <span class=\"text-[11px] font-bold uppercase tracking-wider text-slate-400\">KV Cache Pool</span>\n          <span class=\"text-[10px] font-mono text-slate-400\">GPU Cache Factor</span>\n        </div>\n        <div class=\"flex items-baseline gap-1.5\">\n          <span class=\"text-2xl font-extrabold font-mono text-white tabular-nums\">{{ stats.engine?.kv_cache_usage_pct || 0 }}%</span>\n          <span class=\"text-[11px] text-slate-400\">allocated</span>\n        </div>\n        <div class=\"mt-2 w-full bg-slate-800/90 rounded-full h-1.5 overflow-hidden\">\n          <div class=\"bg-gradient-to-r from-blue-500 to-emerald-400 h-1.5 rounded-full transition-all duration-300\" :style=\"{ width: (stats.engine?.kv_cache_usage_pct || 0) + '%' }\"></div>\n        </div>\n      </div>\n\n      <!-- Card 3: Cache Hit & Concurrency -->\n      <div class=\"glass-card rounded-xl p-3.5 transition-all duration-200\">\n        <div class=\"flex items-center justify-between text-slate-400 mb-1\">\n          <span class=\"text-[11px] font-bold uppercase tracking-wider text-slate-400\">Prefix Hit & MTP</span>\n          <span class=\"text-[10px] font-mono font-semibold\" :class=\"(stats.engine?.running_requests || 0) > 0 ? 'text-emerald-400' : 'text-slate-500'\">\n            {{ stats.engine?.running_requests || 0 }} running\n          </span>\n        </div>\n        <div class=\"flex items-baseline gap-3\">\n          <div>\n            <span class=\"text-xl font-bold font-mono text-white tabular-nums\">{{ stats.engine?.prefix_cache_hit_pct || 0 }}%</span>\n            <span class=\"text-[10px] text-slate-400 font-mono ml-1\">prefix</span>\n          </div>\n          <div class=\"h-6 w-[1px] bg-slate-800\" v-if=\"stats.engine?.spec_acceptance_pct !== null && stats.engine?.spec_acceptance_pct !== undefined\"></div>\n          <div v-if=\"stats.engine?.spec_acceptance_pct !== null && stats.engine?.spec_acceptance_pct !== undefined\">\n            <span class=\"text-xl font-bold font-mono text-emerald-400 tabular-nums\">{{ stats.engine.spec_acceptance_pct }}%</span>\n            <span class=\"text-[10px] text-slate-400 font-mono ml-1\">MTP</span>\n          </div>\n        </div>\n        <div class=\"mt-2 flex items-center justify-between text-[11px] text-slate-400 font-mono pt-2 border-t border-slate-800/80\">\n          <span>Queue state:</span>\n          <span class=\"text-slate-200 tabular-nums\">{{ stats.engine?.waiting_requests || 0 }} waiting in queue</span>\n        </div>\n      </div>\n\n      <!-- Card 4: Hardware VRAM & Temp -->\n      <div class=\"glass-card rounded-xl p-3.5 transition-all duration-200\">\n        <div class=\"flex items-center justify-between text-slate-400 mb-1\">\n          <span class=\"text-[11px] font-bold uppercase tracking-wider text-slate-400\">Hardware VRAM</span>\n          <span class=\"text-[10px] font-mono px-1.5 py-0.5 rounded font-bold\" :class=\"(stats.gpu?.temp_c || 0) > 85 ? 'bg-rose-950 text-rose-400' : 'bg-emerald-950 text-emerald-400'\">\n            {{ stats.gpu?.temp_c || 0 }}°C\n          </span>\n        </div>\n        <div class=\"flex items-baseline gap-1.5\">\n          <span class=\"text-2xl font-extrabold font-mono text-white tabular-nums\">{{ Math.round((stats.gpu?.mem_used_mb || 0)/1024) }}</span>\n          <span class=\"text-[11px] text-slate-400 font-mono tabular-nums\">/ {{ Math.round((stats.gpu?.mem_total_mb || 0)/1024) }} GB</span>\n        </div>\n        <div class=\"mt-2 flex items-center justify-between text-[11px] text-slate-400 font-mono pt-2 border-t border-slate-800/80\">\n          <span>GPU Load / Power:</span>\n          <span class=\"text-slate-200 font-semibold tabular-nums\">{{ stats.gpu?.util_percent || 0 }}% \u2022 {{ Math.round(stats.gpu?.power_draw_w || 0) }}W</span>\n        </div>\n      </div>\n\n    </div>\n\n    <!-- 60FPS STREAMING CANVAS TIMELINE (NO OVERLAY, CLEAN PARALLEL LANES & ZOOM) -->\n    <div class=\"glass-card rounded-xl p-4 relative flex flex-col\">\n      <div class=\"flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80\">\n        <div class=\"flex items-center gap-2\">\n          <div class=\"w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse\"></div>\n          <h2 class=\"text-sm font-bold text-white flex items-center gap-1.5\">\n            Concurrent Request Timeline (Dedicated Parallel Lanes)\n          </h2>\n          <span class=\"text-[11px] font-mono text-emerald-400 bg-emerald-950/80 border border-emerald-800/50 px-2 py-0.5 rounded\">\n            {{ activeLanesCount }} Concurrent Lanes\n          </span>\n        </div>\n\n        <!-- Controls: Time Zoom, Lane Height Density, Container Height & Pause -->\n        <div class=\"flex flex-wrap items-center gap-3 text-xs font-mono\">\n          \n          <!-- Horizontal Time Zoom -->\n          <div class=\"flex items-center gap-1 bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-[11px]\">\n            <span class=\"text-slate-500 px-1.5 text-[10px]\">Time:</span>\n            <button v-for=\"w in [15, 30, 60, 120, 300]\" :key=\"w\" @click=\"timeWindowSec = w\" :class=\"timeWindowSec === w ? 'bg-emerald-600 text-black font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-2 py-0.5 rounded transition\">\n              {{ w < 60 ? w + 's' : (w/60) + 'm' }}\n            </button>\n          </div>\n\n          <!-- Vertical Lane Density (Row Height Zoom) -->\n          <div class=\"flex items-center gap-1 bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-[11px]\">\n            <span class=\"text-slate-500 px-1.5 text-[10px]\">Row:</span>\n            <button @click=\"laneDensity = 14\" :class=\"laneDensity === 14 ? 'bg-slate-700 text-emerald-400 font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-2 py-0.5 rounded transition\" title=\"Tight (14px)\">Tight</button>\n            <button @click=\"laneDensity = 18\" :class=\"laneDensity === 18 ? 'bg-slate-700 text-emerald-400 font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-2 py-0.5 rounded transition\" title=\"Normal (18px)\">Normal</button>\n            <button @click=\"laneDensity = 26\" :class=\"laneDensity === 26 ? 'bg-slate-700 text-emerald-400 font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-2 py-0.5 rounded transition\" title=\"Spacious (26px)\">Comfort</button>\n          </div>\n\n          <!-- Viewport Height Presets -->\n          <div class=\"flex items-center gap-1 bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-[11px]\">\n            <span class=\"text-slate-500 px-1.5 text-[10px]\">View:</span>\n            <button @click=\"viewportHeight = 260\" :class=\"viewportHeight === 260 ? 'bg-slate-700 text-white font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-1.5 py-0.5 rounded\">260px</button>\n            <button @click=\"viewportHeight = 420\" :class=\"viewportHeight === 420 ? 'bg-slate-700 text-white font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-1.5 py-0.5 rounded\">420px</button>\n            <button @click=\"viewportHeight = 600\" :class=\"viewportHeight === 600 ? 'bg-slate-700 text-white font-bold' : 'text-slate-400 hover:text-white'\" class=\"px-1.5 py-0.5 rounded\">Auto</button>\n          </div>\n\n          <!-- Pause/Resume Stream -->\n          <button @click=\"isTimelinePaused = !isTimelinePaused\" :class=\"isTimelinePaused ? 'bg-amber-600 text-white' : 'bg-slate-800 text-slate-300 hover:text-white'\" class=\"px-2.5 py-1 rounded-lg border border-slate-700 text-[11px] flex items-center gap-1.5 transition\">\n            <i :data-lucide=\"isTimelinePaused ? 'play' : 'pause'\" class=\"w-3 h-3\"></i>\n            <span>{{ isTimelinePaused ? 'Resume' : 'Pause' }}</span>\n          </button>\n        </div>\n      </div>\n\n      <!-- Scrollable Canvas Viewport (Never Overlays Rows, Auto-expands) -->\n      <div \n        ref=\"canvasContainer\"\n        class=\"mt-3 relative w-full bg-dark-950/90 rounded-xl border border-slate-800/80 overflow-y-auto overflow-x-hidden select-none\"\n        :style=\"{ height: viewportHeight + 'px' }\"\n      >\n        <canvas \n          ref=\"timelineCanvas\" \n          class=\"w-full block cursor-crosshair\"\n          @mousemove=\"handleCanvasMouseMove\"\n          @mouseleave=\"hoveredReq = null\"\n        ></canvas>\n      </div>\n\n      <!-- Tooltip Popover (Matching screenshot) -->\n      <div \n        v-if=\"hoveredReq\" \n        class=\"absolute z-40 bg-dark-900/95 backdrop-blur-md border border-slate-700 rounded-xl p-3.5 shadow-2xl text-xs font-mono space-y-2 pointer-events-none w-72 transition-all duration-75\"\n        :style=\"tooltipPosStyle\"\n      >\n        <div class=\"flex items-center justify-between pb-2 border-b border-slate-800\">\n          <span class=\"font-bold text-white truncate text-sm\">{{ hoveredReq.model }}</span>\n          <span class=\"px-2 py-0.5 rounded text-[10px] font-bold\" :class=\"getStatusBadgeClass(hoveredReq.status)\">\n            {{ hoveredReq.status === 'active' ? 'ACTIVE' : hoveredReq.status }}\n          </span>\n        </div>\n\n        <div class=\"space-y-1 text-slate-300 text-[11px]\">\n          <div class=\"flex justify-between\">\n            <span class=\"text-slate-500\">Started:</span>\n            <span>{{ formatTime(hoveredReq.start_time) }}</span>\n          </div>\n          <div class=\"flex justify-between\">\n            <span class=\"text-slate-500\">Ended:</span>\n            <span>{{ hoveredReq.end_time ? formatTime(hoveredReq.end_time) : 'In Progress (Active)' }}</span>\n          </div>\n          <div class=\"flex justify-between\">\n            <span class=\"text-slate-500\">Duration:</span>\n            <span class=\"text-emerald-400 font-bold\">{{ formatNumber(hoveredReq.duration_ms) }} ms</span>\n          </div>\n          <div class=\"flex justify-between pt-1 border-t border-slate-800\">\n            <span class=\"text-slate-500\">Tokens (In / Out):</span>\n            <span class=\"text-white font-semibold\">{{ formatNumber(hoveredReq.prompt_tokens || 0) }} / {{ formatNumber(hoveredReq.completion_tokens || 0) }}</span>\n          </div>\n          <div class=\"flex justify-between\" v-if=\"hoveredReq.tps > 0\">\n            <span class=\"text-slate-500\">Speed:</span>\n            <span class=\"text-emerald-400 font-bold\">⚡ {{ hoveredReq.tps }} tok/s</span>\n          </div>\n          <div v-if=\"hoveredReq.prompt_preview\" class=\"pt-1 border-t border-slate-800 text-[10px] text-slate-400 italic truncate\">\n            \"{{ hoveredReq.prompt_preview }}\"\n          </div>\n        </div>\n      </div>\n\n    </div>\n\n    <!-- SEPARATE DUAL THROUGHPUT CHARTS (SPLIT INTO DEDICATED AUTOSCALING RATIOS) -->\n    <div class=\"grid grid-cols-1 lg:grid-cols-2 gap-4\">\n      \n      <!-- Chart 1: Generation Speed (Dedicated 0 .. 300 tok/s with responsive spikes) -->\n      <div class=\"glass-card rounded-xl p-4\">\n        <div class=\"flex flex-wrap items-center justify-between gap-2 mb-2\">\n          <div class=\"flex items-center gap-2\">\n            <i data-lucide=\"zap\" class=\"w-4 h-4 text-emerald-400\"></i>\n            <div>\n              <h2 class=\"text-sm font-bold text-white flex items-center gap-1.5\">\n                Generation Speed Timeline\n                <span class=\"text-[10px] text-slate-400 font-mono font-normal\">(Rolling ~60s)</span>\n              </h2>\n            </div>\n          </div>\n          <div class=\"flex items-center gap-2.5 text-xs font-mono\">\n            <div class=\"flex items-center justify-end gap-1.5\">\n              <span class=\"text-slate-400\">Now:</span>\n              <span class=\"text-emerald-400 font-bold text-sm tabular-nums w-[72px] text-right\">{{ formatTps(stats.engine?.gen_throughput_tps || 0) }} tok/s</span>\n            </div>\n            <div class=\"h-3 w-[1px] bg-slate-800\"></div>\n            <div class=\"flex items-center justify-end gap-1.5\" title=\"Time-weighted average generation speed over the last ~60s\">\n              <span class=\"text-slate-400\">Avg:</span>\n              <span class=\"text-emerald-300 font-semibold tabular-nums w-[72px] text-right\">{{ formatTps(genWindowAvg) }} tok/s</span>\n            </div>\n            <div class=\"h-3 w-[1px] bg-slate-800\"></div>\n            <div class=\"flex items-center justify-end gap-1.5\" title=\"Peak generation speed in this window\">\n              <span class=\"text-slate-400\">Peak:</span>\n              <span class=\"text-white font-bold tabular-nums w-[72px] text-right\">{{ formatTps(genWindowPeak || stats.engine?.peak_gen_tps || stats.engine?.gen_throughput_tps || 0) }} tok/s</span>\n            </div>\n          </div>\n        </div>\n        <div class=\"h-36 w-full\">\n          <canvas id=\"genChart\"></canvas>\n        </div>\n      </div>\n\n      <!-- Chart 2: Prompt Prefill Speed (Dedicated 0 .. 8000+ tok/s ratio) -->\n      <div class=\"glass-card rounded-xl p-4\">\n        <div class=\"flex flex-wrap items-center justify-between gap-2 mb-2\">\n          <div class=\"flex items-center gap-2\">\n            <i data-lucide=\"flame\" class=\"w-4 h-4 text-blue-400\"></i>\n            <div>\n              <h2 class=\"text-sm font-bold text-white flex items-center gap-1.5\">\n                Prompt Prefill Speed Timeline\n                <span class=\"text-[10px] text-slate-400 font-mono font-normal\">(Rolling ~60s)</span>\n              </h2>\n            </div>\n          </div>\n          <div class=\"flex items-center gap-2.5 text-xs font-mono\">\n            <div class=\"flex items-center justify-end gap-1.5\">\n              <span class=\"text-slate-400\">Now:</span>\n              <span class=\"text-blue-400 font-bold text-sm tabular-nums w-[72px] text-right\">{{ formatTps(stats.engine?.prompt_throughput_tps || 0) }} tok/s</span>\n            </div>\n            <div class=\"h-3 w-[1px] bg-slate-800\"></div>\n            <div class=\"flex items-center justify-end gap-1.5\" title=\"Time-weighted average prefill speed during prompt ingestion over the last ~60s\">\n              <span class=\"text-slate-400\">Burst Avg:</span>\n              <span class=\"text-blue-300 font-semibold tabular-nums w-[72px] text-right\">{{ formatTps(prefillWindowAvg || stats.engine?.prompt_throughput_tps || 0) }} tok/s</span>\n            </div>\n            <div class=\"h-3 w-[1px] bg-slate-800\"></div>\n            <div class=\"flex items-center justify-end gap-1.5\" title=\"Peak prefill speed in this window\">\n              <span class=\"text-slate-400\">Peak:</span>\n              <span class=\"text-cyan-300 font-bold tabular-nums w-[72px] text-right\">{{ formatTps(prefillWindowPeak || stats.engine?.peak_prompt_tps || stats.engine?.prompt_throughput_tps || 0) }} tok/s</span>\n            </div>\n          </div>\n        </div>\n        <div class=\"h-36 w-full\">\n          <canvas id=\"prefillChart\"></canvas>\n        </div>\n      </div>\n\n    </div>\n\n    <!-- Main Workspace: Enlarged Logs & Speed Playground -->\n    <div class=\"grid grid-cols-1 lg:grid-cols-12 gap-4\">\n      \n      <!-- Left 7 Cols: Enlarged Live Journal & Request Logs -->\n      <div class=\"lg:col-span-7 space-y-4\">\n        <div class=\"glass-card rounded-xl p-4 flex flex-col h-full\">\n          <div class=\"flex flex-wrap items-center justify-between gap-2 mb-2.5\">\n            <div class=\"flex items-center gap-2\">\n              <i data-lucide=\"terminal\" class=\"w-4 h-4 text-emerald-400\"></i>\n              <h2 class=\"text-sm font-bold text-white\">Live Engine Journal & Stream</h2>\n              <span class=\"text-[11px] font-mono text-slate-500\">({{ filteredLogs.length }} events)</span>\n            </div>\n\n            <!-- Filter Chips & Actions -->\n            <div class=\"flex items-center gap-1.5\">\n              <div class=\"flex items-center bg-slate-900 rounded-lg p-0.5 border border-slate-800 text-[11px] font-mono\">\n                <button @click=\"logFilter = 'all'\" :class=\"logFilter === 'all' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'\" class=\"px-2 py-0.5 rounded\">All</button>\n                <button @click=\"logFilter = 'engine'\" :class=\"logFilter === 'engine' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'\" class=\"px-2 py-0.5 rounded\">Stats</button>\n                <button @click=\"logFilter = 'reqs'\" :class=\"logFilter === 'reqs' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'\" class=\"px-2 py-0.5 rounded\">POSTs</button>\n              </div>\n\n              <button @click=\"copyLogs\" class=\"p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition text-xs\" title=\"Copy Logs\">\n                <i data-lucide=\"copy\" class=\"w-3.5 h-3.5\"></i>\n              </button>\n            </div>\n          </div>\n\n          <!-- The Enlarged Logs Box -->\n          <div ref=\"logBox\" class=\"bg-dark-950/90 rounded-xl p-3 font-mono text-[11px] text-slate-300 border border-slate-800/80 flex-1 min-h-[220px] max-h-72 overflow-y-auto space-y-1\">\n            <div v-if=\"filteredLogs.length === 0\" class=\"text-slate-500 italic py-2 text-center\">No matching log events in buffer...</div>\n            <div v-for=\"(log, idx) in filteredLogs\" :key=\"idx\" class=\"leading-relaxed whitespace-pre-wrap break-all py-0.5\" :class=\"getLogClass(log)\">\n              {{ log }}\n            </div>\n          </div>\n        </div>\n      </div>\n\n      <!-- Right 5 Cols: Speed Playground & Model Management -->\n      <div class=\"lg:col-span-5 space-y-4\">\n        \n        <div class=\"glass-card rounded-xl p-4 flex flex-col h-full\">\n          \n          <!-- Playground Header & Model Dropdown -->\n          <div class=\"flex items-center justify-between gap-2 mb-3 pb-3 border-b border-slate-800\">\n            <div class=\"flex items-center gap-2\">\n              <i data-lucide=\"play-circle\" class=\"w-4 h-4 text-emerald-400\"></i>\n              <h2 class=\"text-sm font-bold text-white\">Interactive Playground</h2>\n            </div>\n            \n            <div class=\"flex items-center gap-1.5\">\n              <span class=\"text-[11px] font-mono text-slate-400\">Model:</span>\n              <select v-model=\"selectedModel\" class=\"bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-xs text-emerald-400 font-mono focus:outline-none focus:border-emerald-500\">\n                <option v-for=\"m in (stats.engine?.models || ['default'])\" :key=\"m\" :value=\"m\">{{ m }}</option>\n              </select>\n            </div>\n          </div>\n\n          <!-- Prompt Form -->\n          <form @submit.prevent=\"runPlayground\" class=\"space-y-3 flex-1 flex flex-col\">\n            <div>\n              <div class=\"flex items-center justify-between mb-1\">\n                <label class=\"text-[11px] font-semibold text-slate-400\">Test Prompt</label>\n                <button type=\"button\" @click=\"promptInput = 'Scrie o functie concisa in Rust pentru streaming HTTP cu reqwest.'\" class=\"text-[10px] text-emerald-400 hover:underline font-mono\">Example prompt</button>\n              </div>\n              <textarea v-model=\"promptInput\" rows=\"2\" class=\"w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500 font-mono resize-none transition\" placeholder=\"Scrie promptul de testare...\"></textarea>\n            </div>\n\n            <!-- Parameters Bar -->\n            <div class=\"flex items-center justify-between text-xs font-mono text-slate-400 bg-slate-900/60 p-2 rounded-lg border border-slate-800/80\">\n              <div class=\"flex items-center gap-3\">\n                <span>Max: <strong class=\"text-slate-200\">128 tok</strong></span>\n                <span>Temp: <strong class=\"text-slate-200\">0.3</strong></span>\n              </div>\n              <button type=\"submit\" :disabled=\"isInferring || !promptInput\" class=\"px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-black font-bold text-xs flex items-center gap-1.5 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-md\">\n                <i data-lucide=\"play\" class=\"w-3 h-3 fill-current\"></i>\n                <span v-if=\"!isInferring\">Run Infer</span>\n                <span v-else>Running...</span>\n              </button>\n            </div>\n\n            <!-- Response & Speed Meter Box -->\n            <div class=\"flex-1 flex flex-col\">\n              <div class=\"flex items-center justify-between text-[11px] text-slate-400 mb-1 font-mono\">\n                <span>Response Output:</span>\n                <span v-if=\"inferStats.totalTokens > 0\" class=\"text-emerald-400 font-bold bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800/50\">\n                  ⚡ {{ inferStats.tps }} tok/s \u2022 {{ inferStats.latencyMs }}ms ({{ inferStats.totalTokens }} tok)\n                </span>\n              </div>\n              <div class=\"w-full flex-1 min-h-[140px] bg-slate-950 border border-slate-800 rounded-xl p-2.5 font-mono text-xs text-slate-200 overflow-y-auto whitespace-pre-wrap leading-relaxed\">\n                <span v-if=\"!inferOutput && !isInferring\" class=\"text-slate-600 italic\">Apasa pe Run Infer pentru a executa modelul live pe H100...</span>\n                <span>{{ inferOutput }}</span>\n              </div>\n            </div>\n\n          </form>\n\n        </div>\n\n      </div>\n\n    </div>\n\n    <!-- cURL / Integration Modal -->\n    <div v-if=\"showCurlModal\" class=\"fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4\">\n      <div class=\"bg-dark-900 border border-slate-700 rounded-2xl max-w-2xl w-full p-5 space-y-4 shadow-2xl\">\n        <div class=\"flex items-center justify-between pb-3 border-b border-slate-800\">\n          <div class=\"flex items-center gap-2\">\n            <i data-lucide=\"terminal\" class=\"w-5 h-5 text-emerald-400\"></i>\n            <h3 class=\"text-base font-bold text-white\">API Integration & cURL Snippets</h3>\n          </div>\n          <button @click=\"showCurlModal = false\" class=\"text-slate-400 hover:text-white p-1\">\n            <i data-lucide=\"x\" class=\"w-5 h-5\"></i>\n          </button>\n        </div>\n\n        <div class=\"bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs font-mono space-y-1\">\n          <div class=\"text-slate-400\">Modele active raportate de <code class=\"text-emerald-400\">/v1/models</code>:</div>\n          <div class=\"flex flex-wrap gap-1.5 pt-1\">\n            <span v-for=\"m in stats.engine?.models\" :key=\"m\" class=\"px-2 py-0.5 rounded bg-slate-800 text-emerald-400 border border-slate-700\">\n              {{ m }}\n            </span>\n          </div>\n        </div>\n\n        <div class=\"space-y-2\">\n          <div class=\"flex items-center justify-between\">\n            <span class=\"text-xs font-bold text-slate-300 uppercase tracking-wider\">cURL Request (OpenAI Compatible)</span>\n            <button @click=\"copyCurlSnippet\" class=\"text-xs text-emerald-400 hover:underline flex items-center gap-1 font-mono\">\n              <i data-lucide=\"copy\" class=\"w-3.5 h-3.5\"></i>\n              <span>{{ copyStatus || 'Copy cURL' }}</span>\n            </button>\n          </div>\n          <pre class=\"bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs font-mono text-emerald-300 overflow-x-auto whitespace-pre leading-relaxed\">{{ curlSnippet }}</pre>\n        </div>\n\n        <div class=\"flex justify-end pt-2\">\n          <button @click=\"showCurlModal = false\" class=\"px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg\">\n            Close\n          </button>\n        </div>\n      </div>\n    </div>\n\n  </div>\n\n  <!-- Vue 3 App Implementation -->\n  <script>\n    const { createApp, ref, computed, onMounted, onUnmounted, nextTick } = Vue;\n\n    createApp({\n      setup() {\n        const stats = ref({});\n        const isRefreshing = ref(false);\n        const logFilter = ref('all');\n        const timelineFilter = ref('all');\n        const timeWindowSec = ref(30); // 30s past window\n        const laneDensity = ref(18); // Default row height (Tight: 14, Normal: 18, Comfort: 26)\n        const viewportHeight = ref(320); // Container height\n        const activeLanesCount = ref(1);\n        const isTimelinePaused = ref(false);\n        const hoveredReq = ref(null);\n        const mouseX = ref(0);\n        const mouseY = ref(0);\n        const timelineCanvas = ref(null);\n        const canvasContainer = ref(null);\n        const selectedModel = ref('');\n        const promptInput = ref(\"Scrie o functie scurta in Python de 2 linii.\");\n        const isInferring = ref(false);\n        const inferOutput = ref(\"\");\n        const inferStats = ref({ totalTokens: 0, tps: 0, latencyMs: 0 });\n        const showCurlModal = ref(false);\n        const copyStatus = ref(\"\");\n        const logBox = ref(null);\n\n        let genChartInstance = null;\n        let prefillChartInstance = null;\n        let animationFrameId = null;\n        const WINDOW_AVG_SECONDS = 60;\n\n        // Plain JS object (Chart.js directly mutates these arrays; making it a\n        // Vue reactive proxy causes infinite recursive observation loops).\n        const chartHistory = {\n          labels: [],\n          t: [],\n          dts: [],\n          genTps: [],\n          promptTps: []\n        };\n\n        // Explicit reactive refs updated in updateCharts on each sample\n        const genWindowAvg = ref(0);\n        const genWindowPeak = ref(0);\n        const prefillWindowAvg = ref(0);\n        const prefillWindowPeak = ref(0);\n\n        const recalculateWindowStats = () => {\n          const now = Date.now();\n          let gsum = 0, gdt = 0, gpeak = 0;\n          let psum = 0, pdt = 0, ppeak = 0;\n          for (let i = 0; i < chartHistory.genTps.length; i++) {\n            const dt = chartHistory.dts[i] || 0;\n            const age = now - (chartHistory.t[i] || now);\n            if (age > WINDOW_AVG_SECONDS * 1000) continue;\n            const g = chartHistory.genTps[i] || 0;\n            const p = chartHistory.promptTps[i] || 0;\n            gsum += g * dt;\n            gdt += dt;\n            if (g > gpeak) gpeak = g;\n            psum += p * dt;\n            pdt += dt;\n            if (p > ppeak) ppeak = p;\n          }\n          genWindowAvg.value = gdt > 0 ? Math.round(gsum / gdt) : 0;\n          genWindowPeak.value = Math.round(gpeak);\n          prefillWindowAvg.value = pdt > 0 ? Math.round(psum / pdt) : 0;\n          prefillWindowPeak.value = Math.round(ppeak);\n        };\n\n        let renderedSpans = [];\n\n        const filteredLogs = computed(() => {\n          const logs = stats.value.recent_logs || [];\n          if (logFilter.value === 'engine') return logs.filter(l => l.includes('Engine 000:'));\n          if (logFilter.value === 'reqs') return logs.filter(l => l.includes('POST /v1/chat/completions'));\n          return logs;\n        });\n\n        const tooltipPosStyle = computed(() => {\n          return {\n            top: `${Math.min(Math.max(mouseY.value + 15, 10), 160)}px`,\n            left: `${Math.min(Math.max(mouseX.value + 15, 10), window.innerWidth - 320)}px`\n          };\n        });\n\n        const curlSnippet = computed(() => {\n          const model = selectedModel.value || stats.value.engine?.active_model || 'default-model';\n          const host = window.location.hostname || '127.0.0.1';\n          return `curl -X POST http://${host}:8000/v1/chat/completions \\\\\n  -H \"Content-Type: application/json\" \\\\\n  -H \"Authorization: Bearer YOUR_API_KEY\" \\\\\n  -d '{\n    \"model\": \"${model}\",\n    \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],\n    \"temperature\": 0.3,\n    \"max_tokens\": 128\n  }'`;\n        });\n\n        // 60FPS Continuous Canvas Animation Loop with No-Overlap Dynamic Lanes\n        const startCanvasLoop = () => {\n          const canvas = timelineCanvas.value;\n          if (!canvas) return;\n          const ctx = canvas.getContext('2d');\n\n          const render = () => {\n            if (!isTimelinePaused.value && !document.hidden) {\n              drawTimeline(canvas, ctx);\n            }\n            animationFrameId = requestAnimationFrame(render);\n          };\n          animationFrameId = requestAnimationFrame(render);\n        };\n\n        const drawTimeline = (canvas, ctx) => {\n          const container = canvasContainer.value;\n          const rect = canvas.getBoundingClientRect();\n          const dpr = window.devicePixelRatio || 1;\n          const width = container ? container.clientWidth : rect.width;\n\n          const now = Date.now() / 1000;\n          const windowSec = timeWindowSec.value;\n          const nowX = width * 0.82; // NOW line at 82%\n          const pps = nowX / windowSec;\n\n          const minTime = now - windowSec;\n          const maxTime = now + (width - nowX) / pps;\n\n          // 1. Fetch and filter timeline requests\n          const allReqs = stats.value.timeline || [];\n          let reqs = allReqs.filter(r => {\n            if (timelineFilter.value === 'active') return r.status === 'active';\n            if (timelineFilter.value === '2xx') return r.status === 200 || r.status === '200';\n            return true;\n          });\n\n          // Sort requests by start_time\n          reqs = reqs.slice().sort((a, b) => a.start_time - b.start_time);\n\n          // 2. Strict Greedy Non-Overlapping Lane Allocation (sub sub sub)\n          const lanes = []; // stores end_time of last request placed in each lane\n          const assignedLanes = [];\n\n          reqs.forEach(req => {\n            let placed = false;\n            for (let i = 0; i < lanes.length; i++) {\n              if (req.start_time >= lanes[i] + 0.05) {\n                lanes[i] = req.end_time || (now + 100);\n                assignedLanes.push(i);\n                placed = true;\n                break;\n              }\n            }\n            if (!placed) {\n              assignedLanes.push(lanes.length);\n              lanes.push(req.end_time || (now + 100));\n            }\n          });\n\n          const totalLanes = Math.max(lanes.length, 1);\n          activeLanesCount.value = totalLanes;\n\n          // Dynamic Lane Height based on density setting\n          const laneHeight = laneDensity.value;\n          const laneGap = laneDensity.value <= 14 ? 3 : 4;\n          const startY = 26;\n\n          // Total required canvas height for all parallel rows\n          const calculatedHeight = Math.max(startY + (totalLanes * (laneHeight + laneGap)) + 20, viewportHeight.value);\n\n          if (canvas.width !== width * dpr || canvas.height !== calculatedHeight * dpr) {\n            canvas.width = width * dpr;\n            canvas.height = calculatedHeight * dpr;\n            canvas.style.height = `${calculatedHeight}px`;\n          }\n\n          ctx.save();\n          ctx.scale(dpr, dpr);\n\n          // Clear background\n          ctx.fillStyle = '#070b14';\n          ctx.fillRect(0, 0, width, calculatedHeight);\n\n          // 3. Draw Alternating Lane Row Backgrounds\n          for (let i = 0; i < totalLanes; i++) {\n            const ly = startY + (i * (laneHeight + laneGap));\n            ctx.fillStyle = i % 2 === 0 ? 'rgba(255, 255, 255, 0.015)' : 'rgba(255, 255, 255, 0.005)';\n            ctx.fillRect(0, ly, width, laneHeight);\n          }\n\n          // 4. Draw Time Grid & Markers\n          ctx.font = '10px \"JetBrains Mono\", monospace';\n          const gridStepSec = windowSec <= 15 ? 2.5 : (windowSec <= 30 ? 5 : (windowSec <= 60 ? 10 : (windowSec <= 120 ? 20 : 60)));\n          const firstGridTime = Math.ceil(minTime / gridStepSec) * gridStepSec;\n\n          for (let t = firstGridTime; t <= maxTime; t += gridStepSec) {\n            const x = nowX - (now - t) * pps;\n            if (x < 0 || x > width) continue;\n\n            const secAgo = Math.round(now - t);\n\n            ctx.strokeStyle = (secAgo === 0) ? 'rgba(16, 185, 129, 0.4)' : 'rgba(255, 255, 255, 0.06)';\n            ctx.lineWidth = 1;\n            ctx.setLineDash(secAgo === 0 ? [3, 2] : []);\n            ctx.beginPath();\n            ctx.moveTo(x, 0);\n            ctx.lineTo(x, calculatedHeight);\n            ctx.stroke();\n            ctx.setLineDash([]);\n\n            if (secAgo !== 0) {\n              ctx.fillStyle = '#64748b';\n              const label = secAgo > 0 ? `-${secAgo}s` : `+${Math.abs(secAgo)}s`;\n              ctx.fillText(label, x + 4, 15);\n            }\n          }\n\n          // 5. Draw Request Spans (Each in its own clean lane!)\n          renderedSpans = [];\n\n          reqs.forEach((req, idx) => {\n            const laneIdx = assignedLanes[idx] || 0;\n            const y = startY + (laneIdx * (laneHeight + laneGap));\n\n            const reqStart = req.start_time;\n            const reqEnd = req.end_time || now;\n\n            const x1 = nowX - (now - reqStart) * pps;\n            const x2 = req.end_time ? (nowX - (now - reqEnd) * pps) : nowX;\n            const spanW = Math.max(x2 - x1, 5);\n\n            if (x2 < -50 || x1 > width + 50) return;\n\n            const isHovered = hoveredReq.value && (hoveredReq.value.id === req.id);\n            const isActive = req.status === 'active';\n\n            // Bar Gradient\n            let grad = ctx.createLinearGradient(x1, y, x2, y);\n            if (isActive) {\n              grad.addColorStop(0, 'rgba(147, 51, 234, 0.95)');\n              grad.addColorStop(1, 'rgba(99, 102, 241, 0.95)');\n            } else if (req.status === 200 || req.status === '200') {\n              grad.addColorStop(0, 'rgba(16, 185, 129, 0.9)');\n              grad.addColorStop(1, 'rgba(20, 184, 166, 0.85)');\n            } else {\n              grad.addColorStop(0, 'rgba(225, 29, 72, 0.9)');\n              grad.addColorStop(1, 'rgba(239, 68, 68, 0.85)');\n            }\n\n            const radius = laneHeight <= 14 ? 2.5 : 4;\n            ctx.fillStyle = grad;\n            ctx.beginPath();\n            ctx.roundRect(x1, y, spanW, laneHeight, radius);\n            ctx.fill();\n\n            // Border\n            if (isActive) {\n              ctx.strokeStyle = '#c084fc';\n              ctx.lineWidth = 1.5;\n              ctx.shadowColor = 'rgba(192, 132, 252, 0.6)';\n              ctx.shadowBlur = 6;\n              ctx.stroke();\n              ctx.shadowBlur = 0;\n            } else if (isHovered) {\n              ctx.strokeStyle = '#ffffff';\n              ctx.lineWidth = 1.5;\n              ctx.stroke();\n            } else {\n              ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';\n              ctx.lineWidth = 1;\n              ctx.stroke();\n            }\n\n            // Text Label if wide enough\n            if (spanW > 30) {\n              ctx.fillStyle = '#ffffff';\n              const fontSize = laneHeight <= 14 ? 9 : (laneHeight <= 18 ? 10 : 11);\n              ctx.font = `bold ${fontSize}px \"JetBrains Mono\", monospace`;\n              const modelLabel = req.model || stats.value.engine?.active_model || 'model';\n              const tokLabel = (req.prompt_tokens && spanW > 70) ? ` (${formatNumber(req.prompt_tokens)}/${req.completion_tokens}t)` : '';\n              const durLabel = spanW > 50 ? ` \u2022 ${req.duration_ms}ms` : '';\n              const fullText = modelLabel + tokLabel + durLabel;\n\n              ctx.save();\n              ctx.beginPath();\n              ctx.rect(x1 + 3, y, spanW - 6, laneHeight);\n              ctx.clip();\n              ctx.fillText(fullText, x1 + 4, y + (laneHeight * 0.72));\n              ctx.restore();\n            }\n\n            // Save hit target for mouse events\n            renderedSpans.push({\n              req,\n              rect: { x: x1, y, width: spanW, height: laneHeight }\n            });\n          });\n\n          // 6. Draw NOW Guideline at 82%\n          ctx.strokeStyle = '#10b981';\n          ctx.lineWidth = 2;\n          ctx.shadowColor = 'rgba(16, 185, 129, 0.8)';\n          ctx.shadowBlur = 8;\n          ctx.beginPath();\n          ctx.moveTo(nowX, 0);\n          ctx.lineTo(nowX, calculatedHeight);\n          ctx.stroke();\n          ctx.shadowBlur = 0;\n\n          // \"NOW |\" Badge at top of line\n          ctx.fillStyle = '#10b981';\n          ctx.beginPath();\n          ctx.roundRect(nowX - 22, 2, 44, 17, 3);\n          ctx.fill();\n\n          ctx.fillStyle = '#000000';\n          ctx.font = 'bold 10px \"JetBrains Mono\", monospace';\n          ctx.fillText('NOW |', nowX - 16, 14);\n\n          ctx.restore();\n        };\n\n        const handleCanvasMouseMove = (e) => {\n          const canvas = timelineCanvas.value;\n          if (!canvas) return;\n          const rect = canvas.getBoundingClientRect();\n          mouseX.value = e.clientX - rect.left;\n          mouseY.value = e.clientY - rect.top;\n\n          const match = renderedSpans.find(s => \n            mouseX.value >= s.rect.x && mouseX.value <= (s.rect.x + s.rect.width) &&\n            mouseY.value >= s.rect.y && mouseY.value <= (s.rect.y + s.rect.height)\n          );\n\n          hoveredReq.value = match ? match.req : null;\n        };\n\n        const initCharts = () => {\n          // 1. Generation Speed Chart (0 .. 300+ tok/s dedicated scale)\n          const genCtx = document.getElementById('genChart');\n          if (genCtx) {\n            genChartInstance = new Chart(genCtx, {\n              type: 'line',\n              data: {\n                labels: chartHistory.labels,\n                datasets: [{\n                  label: 'Generation (tok/s)',\n                  data: chartHistory.genTps,\n                  borderColor: '#10b981',\n                  backgroundColor: 'rgba(16, 185, 129, 0.15)',\n                  borderWidth: 2,\n                  fill: true,\n                  tension: 0.35,\n                  pointRadius: 0\n                }]\n              },\n              options: {\n                responsive: true,\n                maintainAspectRatio: false,\n                animation: false,\n                scales: {\n                  x: {\n                    grid: { color: 'rgba(255, 255, 255, 0.04)' },\n                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }\n                  },\n                  y: {\n                    beginAtZero: true,\n                    suggestedMax: 200,\n                    grid: { color: 'rgba(255, 255, 255, 0.04)' },\n                    ticks: { color: '#10b981', font: { family: 'JetBrains Mono', size: 9 } }\n                  }\n                },\n                plugins: { legend: { display: false } }\n              }\n            });\n          }\n\n          // 2. Prefill Speed Chart (0 .. 8000+ tok/s dedicated scale)\n          const prefillCtx = document.getElementById('prefillChart');\n          if (prefillCtx) {\n            prefillChartInstance = new Chart(prefillCtx, {\n              type: 'line',\n              data: {\n                labels: chartHistory.labels,\n                datasets: [{\n                  label: 'Prefill (tok/s)',\n                  data: chartHistory.promptTps,\n                  borderColor: '#3b82f6',\n                  backgroundColor: 'rgba(59, 130, 246, 0.12)',\n                  borderWidth: 2,\n                  fill: true,\n                  tension: 0.35,\n                  pointRadius: 0\n                }]\n              },\n              options: {\n                responsive: true,\n                maintainAspectRatio: false,\n                animation: false,\n                scales: {\n                  x: {\n                    grid: { color: 'rgba(255, 255, 255, 0.04)' },\n                    ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }\n                  },\n                  y: {\n                    beginAtZero: true,\n                    suggestedMax: 3000,\n                    grid: { color: 'rgba(255, 255, 255, 0.04)' },\n                    ticks: { color: '#3b82f6', font: { family: 'JetBrains Mono', size: 9 } }\n                  }\n                },\n                plugins: { legend: { display: false } }\n              }\n            });\n          }\n        };\n\n        let lastSampleMs = 0;\n\n        const updateCharts = (genTps, promptTps) => {\n          const nowMs = Date.now();\n          const nowLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });\n          const dt = lastSampleMs > 0 ? (nowMs - lastSampleMs) / 1000 : 1;\n          lastSampleMs = nowMs;\n\n          chartHistory.labels.push(nowLabel);\n          chartHistory.t.push(nowMs);\n          chartHistory.dts.push(dt);\n          chartHistory.genTps.push(genTps);\n          chartHistory.promptTps.push(promptTps);\n\n          const max = 90;\n          if (chartHistory.labels.length > max) {\n            chartHistory.labels.shift();\n            chartHistory.t.shift();\n            chartHistory.dts.shift();\n            chartHistory.genTps.shift();\n            chartHistory.promptTps.shift();\n          }\n\n          recalculateWindowStats();\n\n          if (genChartInstance) genChartInstance.update();\n          if (prefillChartInstance) prefillChartInstance.update();\n        };\n\n        let lastStatsSuccess = Date.now();\n\n        const fetchStats = async (manual = false) => {\n          if (!manual && document.hidden) return;\n          if (manual) isRefreshing.value = true;\n          try {\n            const res = await fetch('/api/stats');\n            if (res.ok) {\n              const data = await res.json();\n              stats.value = data;\n              lastStatsSuccess = Date.now();\n              if (data.engine?.models?.length && !selectedModel.value) {\n                selectedModel.value = data.engine.models[0];\n              }\n              updateCharts(data.engine?.gen_throughput_tps || 0, data.engine?.prompt_throughput_tps || 0);\n            }\n          } catch (e) {\n            console.error(\"Stats fetch error:\", e);\n          } finally {\n            if (manual) isRefreshing.value = false;\n            nextTick(() => lucide.createIcons());\n          }\n        };\n\n        const isStatsStale = () => (Date.now() - lastStatsSuccess) > 5000;\n\n        const runPlayground = async () => {\n          if (!promptInput.value || isInferring.value) return;\n          isInferring.value = true;\n          inferOutput.value = \"\";\n          inferStats.value = { totalTokens: 0, tps: 0, latencyMs: 0 };\n          const startTime = performance.now();\n\n          try {\n            const res = await fetch('/api/chat', {\n              method: 'POST',\n              headers: { 'Content-Type': 'application/json' },\n              body: JSON.stringify({\n                model: selectedModel.value || stats.value.engine?.active_model || 'default',\n                messages: [{ role: 'user', content: promptInput.value }],\n                max_tokens: 128,\n                temperature: 0.3\n              })\n            });\n\n            if (res.ok) {\n              const data = await res.json();\n              const elapsedSec = (performance.now() - startTime) / 1000;\n              const content = data.choices?.[0]?.message?.content || \"\";\n              const completionTokens = data.usage?.completion_tokens || 0;\n              inferOutput.value = content;\n              inferStats.value = {\n                totalTokens: completionTokens,\n                tps: Math.round(completionTokens / Math.max(elapsedSec, 0.01)),\n                latencyMs: Math.round(elapsedSec * 1000)\n              };\n            } else {\n              inferOutput.value = `[Error ${res.status}]: ${await res.text()}`;\n            }\n          } catch (e) {\n            inferOutput.value = `[Request Failed]: ${e.message}`;\n          } finally {\n            isInferring.value = false;\n            fetchStats();\n          }\n        };\n\n        const copyLogs = () => {\n          const text = (stats.value.recent_logs || []).join(\"\\n\");\n          navigator.clipboard.writeText(text);\n        };\n\n        const copyCurlSnippet = () => {\n          navigator.clipboard.writeText(curlSnippet.value);\n          copyStatus.value = \"Copied!\";\n          setTimeout(() => { copyStatus.value = \"\"; }, 2000);\n        };\n\n        const formatNumber = (num) => new Intl.NumberFormat().format(num || 0);\n\n        const formatCompactNumber = (num) => {\n          if (!num) return '0';\n          if (num >= 1_000_000_000) return (num / 1_000_000_000).toFixed(1) + 'B';\n          if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';\n          if (num >= 1_000) return (num / 1_000).toFixed(1) + 'k';\n          return num.toString();\n        };\n\n        // Fixed-width variants (up to 3 digits before the dot, always 1 decimal) so\n        // values never change width as digits tick over and push the layout around.\n        const formatTps = (num) => {\n          if (!num || num <= 0) return '0.0';\n          if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';\n          if (num >= 1_000) return (num / 1_000).toFixed(1) + 'k';\n          return num.toFixed(1);\n        };\n\n        const formatMs = (num) => {\n          if (num == null || num <= 0) return '--';\n          if (num >= 60000) return (num / 60000).toFixed(1) + 'm';\n          if (num >= 1000) return (num / 1000).toFixed(1) + 's';\n          return Math.round(num).toLocaleString() + 'ms';\n        };\n\n        const formatTime = (epochSec) => {\n          if (!epochSec) return '-';\n          return new Date(epochSec * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });\n        };\n\n        const getStatusBadgeClass = (status) => {\n          if (status === 'active') return 'bg-purple-950 text-purple-300 border border-purple-800';\n          if (status === 200 || status === '200') return 'bg-emerald-950 text-emerald-300 border border-emerald-800';\n          return 'bg-rose-950 text-rose-300 border border-rose-800';\n        };\n\n        const getLogClass = (log) => {\n          if (log.includes('Engine 000:')) return 'text-emerald-400 font-semibold';\n          if (log.includes('SpecDecoding')) return 'text-blue-400';\n          if (log.includes('POST /v1/chat/completions')) return 'text-slate-300';\n          if (log.includes('ERROR') || log.includes('Exception')) return 'text-rose-400';\n          return 'text-slate-400';\n        };\n\n        onMounted(() => {\n          initCharts();\n          fetchStats();\n          startCanvasLoop();\n          window._statsInterval = setInterval(fetchStats, 2000);\n          lucide.createIcons();\n        });\n\n        onUnmounted(() => {\n          if (animationFrameId) cancelAnimationFrame(animationFrameId);\n          if (window._statsInterval) clearInterval(window._statsInterval);\n        });\n\n        return {\n          stats,\n          isRefreshing,\n          logFilter,\n          timelineFilter,\n          timeWindowSec,\n          laneDensity,\n          viewportHeight,\n          activeLanesCount,\n          isTimelinePaused,\n          hoveredReq,\n          timelineCanvas,\n          canvasContainer,\n          filteredLogs,\n          selectedModel,\n          promptInput,\n          isInferring,\n          inferOutput,\n          inferStats,\n          showCurlModal,\n          curlSnippet,\n          copyStatus,\n          logBox,\n          tooltipPosStyle,\n          handleCanvasMouseMove,\n          fetchStats,\n          isStatsStale,\n          runPlayground,\n          copyLogs,\n          copyCurlSnippet,\n          formatNumber,\n          formatCompactNumber,\n          formatTps,\n          formatMs,\n          formatTime,\n          getStatusBadgeClass,\n          getLogClass,\n          genWindowAvg,\n          genWindowPeak,\n          prefillWindowAvg,\n          prefillWindowPeak\n        };\n      }\n    }).mount('#app');\n  </script>\n</body>\n</html>\n"

class VLLMProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        if STATIC_DIR and STATIC_DIR.exists():
            super().__init__(*args, directory=str(STATIC_DIR), **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/stats":
            stats = get_stats()
            body = json.dumps(stats).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif path in ("/api/models", "/v1/models"):
            models = get_models_list()
            res = {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "16")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        elif path in ("/", "/index.html"):
            if STATIC_DIR and (STATIC_DIR / "index.html").exists():
                return super().do_GET()
            body = EMBEDDED_INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        else:
            if STATIC_DIR and (STATIC_DIR / path.lstrip("/")).exists():
                return super().do_GET()
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _active_proxy_count
        path = self.path.split("?")[0]
        if path in ("/v1/chat/completions", "/api/chat/completions", "/api/chat"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > MAX_REQUEST_BODY_BYTES:
                self.send_response(413)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Request body too large"}).encode())
                return
            post_data = self.rfile.read(content_length)

            model_name = "default"
            prompt_preview = ""
            try:
                body_json = json.loads(post_data.decode("utf-8"))
                model_name = body_json.get("model", "default")
                msgs = body_json.get("messages", [])
                if msgs and isinstance(msgs, list):
                    prompt_preview = str(msgs[-1].get("content", ""))
            except Exception:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode())
                return

            req_id = tracker.start_request(model=model_name, prompt=prompt_preview)

            # Concurrency guard
            with _proxy_lock:
                if _active_proxy_count >= MAX_CONCURRENT_REQUESTS:
                    tracker.finish_request(req_id, status_code=503, model=model_name)
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Server busy, try again"}).encode())
                    return
                _active_proxy_count += 1

            try:
                target_url = f"{Config.vllm_url.rstrip('/')}/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                if Config.api_key:
                    headers["Authorization"] = f"Bearer {Config.api_key}"
                
                req = urllib.request.Request(target_url, data=post_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp_data = resp.read()
                    status = resp.status
                    
                    p_tok, c_tok = 0, 0
                    try:
                        resp_json = json.loads(resp_data.decode("utf-8"))
                        usage = resp_json.get("usage", {})
                        p_tok = usage.get("prompt_tokens", 0)
                        c_tok = usage.get("completion_tokens", 0)
                    except Exception:
                        pass

                    tracker.finish_request(req_id, status_code=status, prompt_tokens=p_tok, completion_tokens=c_tok, model=model_name)

                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
            except urllib.error.HTTPError as e:
                err_data = e.read()
                tracker.finish_request(req_id, status_code=e.code, model=model_name)
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
            except Exception as e:
                tracker.finish_request(req_id, status_code=500, model=model_name)
                msg = json.dumps({"error": "Internal server error"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            finally:
                with _proxy_lock:
                    _active_proxy_count -= 1
            return
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "http://" + self.headers.get("Host", "127.0.0.1:8080"))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def main():
    parser = argparse.ArgumentParser(description="vLLM Vue3 Telemetry & Control Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind to (default: {DEFAULT_PORT})")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Host to bind to (default: {DEFAULT_HOST})")
    parser.add_argument("--vllm-url", type=str, default=DEFAULT_VLLM_URL, help=f"vLLM base URL (default: {DEFAULT_VLLM_URL})")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="vLLM API Key (optional)")
    args = parser.parse_args()

    Config.port = args.port
    Config.host = args.host
    Config.vllm_url = args.vllm_url
    Config.api_key = args.api_key

    server_address = (Config.host, Config.port)
    start_samplers()
    httpd = ThreadingHTTPServer(server_address, VLLMProxyHandler)
    print(f"==================================================")
    print(f"🚀 vLLM Telemetry Dashboard running at http://{Config.host}:{Config.port}")
    print(f"📡 Monitoring vLLM Backend at {Config.vllm_url}")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
