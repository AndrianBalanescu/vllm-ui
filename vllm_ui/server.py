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
from pathlib import Path

# Defaults
DEFAULT_PORT = 8080
DEFAULT_HOST = "0.0.0.0"
DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "supersecretdev")

STATIC_DIR = Path(__file__).parent / "static"

class Config:
    port = DEFAULT_PORT
    host = DEFAULT_HOST
    vllm_url = DEFAULT_VLLM_URL
    api_key = DEFAULT_API_KEY

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
                for lmatch in re.finditer(r'([a-zA-Z_0-9]+)="([^"]*)"', labels_str):
                    labels[lmatch.group(1)] = lmatch.group(2)
                metrics[name].append({"labels": labels, "value": val})
            else:
                metrics[name] = val
    return metrics

def get_recent_vllm_logs():
    try:
        out = subprocess.check_output(
            ["sudo", "journalctl", "-u", "vllm", "-n", "80", "--no-pager"],
            stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        lines = []
        for l in out.splitlines():
            if "Engine 000:" in l or "SpecDecoding metrics:" in l or "POST /v1/chat/completions" in l or "ERROR" in l or "INFO:" in l:
                lines.append(l.strip())
        return lines[-40:]
    except Exception:
        return []

def get_models_list():
    try:
        url = f"{Config.vllm_url.rstrip('/')}/v1/models"
        headers = {}
        if Config.api_key:
            headers["Authorization"] = f"Bearer {Config.api_key}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["id"] for m in data.get("data", []) if not m.get("id", "").startswith("modelperm-")]
    except Exception:
        return ["qwen3.8-27b-clean-int4"]

def check_service_health(port):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False

def get_stats():
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
    
    def get_single(key, default=0.0):
        val = parsed.get(key, default)
        if isinstance(val, list) and len(val) > 0:
            return val[0]["value"]
        return float(val) if isinstance(val, (int, float)) else default

    running = get_single("vllm:num_requests_running", 0)
    waiting = get_single("vllm:num_requests_waiting", 0)
    kv_cache_usage = get_single("vllm:gpu_cache_usage_factor", 0) * 100.0
    prompt_throughput = get_single("vllm:avg_prompt_throughput_tok_per_s", 0)
    gen_throughput = get_single("vllm:avg_generation_throughput_tok_per_s", 0)
    
    accepted_drafts = get_single("vllm:spec_decode_num_accepted_tokens_total", 0)
    total_drafts = get_single("vllm:spec_decode_num_draft_tokens_total", 0)
    spec_acceptance_pct = (accepted_drafts / total_drafts * 100.0) if total_drafts > 0 else 85.0
    
    cached_tokens = get_single("vllm:num_cached_tokens_total", 0)
    total_prompt_tokens = get_single("vllm:prompt_tokens_total", 0)
    prefix_hit_pct = (cached_tokens / (cached_tokens + total_prompt_tokens) * 100.0) if (cached_tokens + total_prompt_tokens) > 0 else 0.0

    logs = get_recent_vllm_logs()
    instant_prompt_tps = prompt_throughput
    instant_gen_tps = gen_throughput
    instant_kv_pct = kv_cache_usage
    instant_prefix_pct = prefix_hit_pct
    
    for l in reversed(logs):
        if "Engine 000:" in l:
            m_p = re.search(r"Avg prompt throughput:\s*([\d.]+)", l)
            m_g = re.search(r"Avg generation throughput:\s*([\d.]+)", l)
            m_kv = re.search(r"GPU KV cache usage:\s*([\d.]+)%", l)
            m_pre = re.search(r"Prefix cache hit rate:\s*([\d.]+)%", l)
            if m_p: instant_prompt_tps = float(m_p.group(1))
            if m_g: instant_gen_tps = float(m_g.group(1))
            if m_kv: instant_kv_pct = float(m_kv.group(1))
            if m_pre: instant_prefix_pct = float(m_pre.group(1))
            break

    models = get_models_list()
    fastembed_ok = check_service_health(8002)
    ibrowse_ok = check_service_health(3000)

    return {
        "timestamp": time.time(),
        "gpu": gpu,
        "services": {
            "vllm": "online" if metrics_raw else "offline",
            "fastembed_cpu": "online" if fastembed_ok else "offline",
            "ibrowse": "online" if ibrowse_ok else "offline"
        },
        "engine": {
            "status": "online" if metrics_raw else "offline",
            "running_requests": int(running),
            "waiting_requests": int(waiting),
            "gen_throughput_tps": round(instant_gen_tps, 1),
            "prompt_throughput_tps": round(instant_prompt_tps, 1),
            "kv_cache_usage_pct": round(instant_kv_pct, 1),
            "prefix_cache_hit_pct": round(instant_prefix_pct, 1),
            "spec_acceptance_pct": round(spec_acceptance_pct, 1),
            "total_kv_slots": 1554888,
            "used_kv_slots": int(1554888 * (instant_kv_pct / 100.0)),
            "models": models,
            "active_model": models[0] if models else "default",
            "max_model_len": 262144
        },
        "recent_logs": logs
    }

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/stats":
            stats = get_stats()
            body = json.dumps(stats).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif self.path == "/api/models" or self.path == "/v1/models":
            models = get_models_list()
            res = {"object": "list", "data": [{"id": m, "object": "model"} for m in models]}
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        elif self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path == "/api/chat" or self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            chat_url = f"{Config.vllm_url.rstrip('/')}/v1/chat/completions"
            
            headers = {"Content-Type": "application/json"}
            if Config.api_key:
                headers["Authorization"] = f"Bearer {Config.api_key}"

            try:
                req = urllib.request.Request(chat_url, data=post_data, headers=headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_data = resp.read()
                    self.send_response(resp.status)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
            except urllib.error.HTTPError as e:
                err_data = e.read()
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
            except Exception as e:
                msg = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            return
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def main():
    parser = argparse.ArgumentParser(description="vLLM Vue3 Telemetry & Control Dashboard")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help=f"Dashboard port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", "-H", type=str, default=DEFAULT_HOST, help=f"Dashboard host (default: {DEFAULT_HOST})")
    parser.add_argument("--vllm-url", "-u", type=str, default=DEFAULT_VLLM_URL, help=f"vLLM base URL (default: {DEFAULT_VLLM_URL})")
    parser.add_argument("--api-key", "-k", type=str, default=DEFAULT_API_KEY, help="vLLM API Key (if required)")
    args = parser.parse_args()

    Config.port = args.port
    Config.host = args.host
    Config.vllm_url = args.vllm_url
    Config.api_key = args.api_key

    print(f"==================================================")
    print(f"🚀 vLLM Web Dashboard")
    print(f"   Dashboard UI : http://{Config.host}:{Config.port}")
    print(f"   Target vLLM  : {Config.vllm_url}")
    print(f"==================================================")

    with ThreadingHTTPServer((Config.host, Config.port), DashboardHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
            sys.exit(0)

if __name__ == "__main__":
    main()
