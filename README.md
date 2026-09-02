# ⚡ vLLM UI & Telemetry Dashboard

Zero-dependency, real-time **Vue 3 Web UI & Telemetry Dashboard** for **vLLM** and local LLM inference engines.

Provides live GPU utilization, token throughput timelines (Prompt & Generation tok/s), KV Cache memory pool visualizer, Prefix Cache hit rates, Speculative Decoding (MTP) metrics, live engine journal logs, and an interactive prompt playground.

---

## 🚀 1-Command Instant Run

### Option 1: Run with `npx` (Zero install)
```bash
npx vllm-ui
```

### Option 2: Run with `pip` (Python)
```bash
pip install vllm-ui
vllm-ui --port 8080 --vllm-url http://127.0.0.1:8000
```

### Option 3: Standalone 1-Liner (cURL)
```bash
curl -fsSL https://raw.githubusercontent.com/AndrianBalanescu/vllm-ui/main/vllm_ui/server.py | python3 -
```

---

## 🌟 Key Features

- ⚡ **Real-Time Throughput Charts**: Live interactive timeline showing generation and prefill speeds (tok/s) sampled continuously.
- 🧠 **KV Cache Memory Pool**: Visual progress bar showing allocated vs. total KV token slots (e.g. 1.55M tokens).
- 🎯 **Prefix & Speculative Decoding**: Real-time tracking of Prefix Cache hit rates and Multi-Token Prediction (MTP) draft acceptance rates.
- 🖥️ **NVIDIA GPU Telemetry**: Live VRAM allocation, GPU core load %, temperature (°C), and power draw (W).
- 📜 **Enlarged Engine Journal**: Live auto-scrolling log stream with quick filters (`All`, `Stats`, `POSTs`) and 1-click clipboard export.
- 🧪 **Interactive Speed Playground**: Test inference latency, time-to-first-token (TTFT), and single-stream tok/s speedometers directly in the browser.
- 📋 **1-Click cURL & OpenAI API Snippets**: Auto-generates ready-to-run cURL commands for any active model in `/v1/models`.

---

## 🛠️ CLI Options

```text
usage: vllm-ui [-h] [--port PORT] [--host HOST] [--vllm-url VLLM_URL] [--api-key API_KEY]

options:
  -h, --help            show this help message and exit
  --port PORT, -p PORT  Dashboard port (default: 8080)
  --host HOST, -H HOST  Dashboard host (default: 0.0.0.0)
  --vllm-url VLLM_URL   vLLM base URL (default: http://127.0.0.1:8000)
  --api-key API_KEY     vLLM API Key (if required)
```

---

## 📄 License
MIT © Andrian Balanescu
