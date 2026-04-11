# RAG Poisoning Project

Defensive research project for secure Retrieval-Augmented Generation (RAG) under data poisoning pressure. This repository contains a working, modular implementation with:

- ingestion and ChromaDB indexing,
- poisoning attack simulation,
- layered retrieval defenses,
- orchestration for secure answer generation,
- benchmark and experiment tooling.

This README is aligned with the current codebase and CLI behavior.

---

## 1. What this project implements

The code models poisoning as an attack on retrieval context, not model weights.

Core workflow:

1. Build or load document corpus.
2. Index into vector store.
3. Inject poisoned artifacts.
4. Retrieve candidates for a question.
5. Apply defenses before generation.
6. Generate answer with local or remote provider.
7. Measure contamination, ASR, and defense effectiveness.

---

## 2. Current architecture

The repository is organized around package-first modules.

### Runtime pipeline

- `src/main.py`:
  - CLI entrypoint for secure RAG run.
  - Supports provider selection and strict defense mode.
- `src/ingestion/`:
  - file-system document source and ingestion pipeline.
- `src/retrieval/`:
  - embedding providers,
  - Chroma-backed vector store,
  - production vector store manager,
  - secure retriever abstraction.
- `src/llm/`:
  - provider implementations (local llama.cpp, Ollama, DeepSeek),
  - prompt hardening,
  - response generator,
  - provider factory.
- `src/defenses/`:
  - provenance filtering,
  - injection pattern defenses,
  - chunk sanitizer,
  - security reranker,
  - diversity enforcer,
  - context firewall,
  - provenance tracker.
- `src/evaluation/`:
  - metrics and benchmark framework.

### Orchestration and experimentation

- `src/secure_rag_orchestrator.py`:
  - defense-first orchestration with 8 stages:
    1. raw retrieval,
    2. provenance filtering,
    3. security reranking,
    4. diversity enforcement,
    5. chunk sanitization,
    6. context firewall,
    7. prompt hardening,
    8. LLM generation.
- `src/attack_simulation_framework.py`:
  - attack generation and injection flow.
  - supported attack types:
    - `prompt_injection`
    - `keyword_stuffing`
    - `targeted_misinformation`
    - `embedding_manipulation_approx`
- `src/experiment_runner.py`:
  - baseline vs defended benchmark execution,
  - CSV/JSON artifact export,
  - optional plot generation.

### Setup and diagnostics

- `setup.sh`:
  - bootstraps virtual environment with `uv`, installs dependencies,
  - downloads embedding model,
  - optionally downloads local GGUF model.
- `test_setup.py`:
  - validates environment, embeddings, Chroma operations, and end-to-end setup checks.
- `src/setup_checks.py`:
  - setup-only helper logic, intentionally separated from runtime utils.

---

## 3. Requirements

- Python 3.11 recommended.
- `uv` required by `setup.sh`.
- Optional inference backends:
  - local GGUF model for llama.cpp path,
  - Ollama server for `--infer ollama`,
  - DeepSeek API key for `--infer deepseek`.

---

## 4. Quick start

### Option A: one-command setup script (recommended)

```bash
chmod +x setup.sh
./setup.sh --no-local
```

Use `--no-local` when you do not want to download the local LLM model.

### Option B: manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Configuration

Configuration is loaded from environment variables, `.env`, and optional `.keys`.

Key variables used by the code:

- `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `VECTOR_DB_PATH` (default: `./data/chroma_db`)
- `LLAMA_MODEL_PATH` (default: `./models/llm/Phi-3.5-mini-instruct.Q4_K_M.gguf`)
- `SENTENCE_TRANSFORMERS_HOME` (default: `./models/embedding`)
- `TRANSFORMERS_CACHE` (default: `./models/embedding`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `Phi-3.5-mini-instruct:latest`)
- `DEEPSEEK_MODEL` (default: `deepseek-chat`)
- `DEEPSEEK_API_KEY` (default: empty)
- `TOP_K_RETRIEVAL` (default: `3`)
- `SIMILARITY_THRESHOLD` (default: `0.0`)
- `LOG_LEVEL` (default: `WARN`)
- `LOG_FILE` (default: `./logs/rag_demo.log`)

Telemetry note:

- Chroma anonymized telemetry is disabled by default in code and noisy telemetry logger output is suppressed.

---

## 6. CLI reference

### 6.1 Setup verification

```bash
python3 test_setup.py --no-local
```

What it checks:

- config load,
- embedding model availability and encoding,
- Chroma read/write/query,
- end-to-end baseline and defended poisoning behavior.

### 6.2 Secure pipeline run

```bash
python3 src/main.py
python3 src/main.py --question "What is our vacation policy?"
python3 src/main.py --infer ollama
python3 src/main.py --infer deepseek
python3 src/main.py --strict
```

CLI flags:

- `--question`: query string.
- `--infer`: one of `local`, `ollama`, `deepseek`.
- `--strict`: enables stricter provenance filtering.
- `--no-local-fallback`: fail fast for local provider if model load/generation fails.

Strict local behavior example:

```bash
python3 src/main.py --infer local --no-local-fallback
```

Without `--no-local-fallback`, missing local model falls back to deterministic fallback text.

### 6.3 Experiment runner

```bash
python3 src/experiment_runner.py --attacks prompt_injection --no-plot
python3 src/experiment_runner.py --attacks prompt_injection,keyword_stuffing --llm-provider ollama
python3 src/experiment_runner.py --attacks targeted_misinformation,embedding_manipulation_approx
```

CLI flags:

- `--attacks`: comma-separated attack names.
- `--top-k`: retrieval depth for benchmark (default `5`).
- `--output-dir`: artifact directory (default `./data/experiment_results`).
- `--llm-provider`: `local`, `ollama`, `deepseek`.
- `--no-local-fallback`: strict fail-fast local mode in defended path.
- `--no-plot`: skip PNG generation.

Experiment strict local example:

```bash
python3 src/experiment_runner.py --attacks prompt_injection --no-local-fallback --no-plot
```

If local model is missing and strict mode is enabled, benchmark exits non-zero.

---

## 7. Outputs and artifacts

Generated files are written under `data/`:

- vector DB persistence: `data/chroma_db/`
- sample corpus:
  - `data/legitimate_documents/`
  - `data/poisoned_documents/`
- experiment artifacts:
  - `data/experiment_results/experiment_metrics_<timestamp>.csv`
  - `data/experiment_results/experiment_report_<timestamp>.json`
  - optional `data/experiment_results/attack_success_vs_defense_<timestamp>.png`

---

## 8. Interpreting results

Important reported signals:

- retrieval contamination rate,
- baseline ASR vs defended ASR,
- defense effectiveness,
- latency overhead,
- precision/recall-like defense metrics,
- diversity score changes.

Typical expected pattern in successful defense run:

- baseline contamination/ASR higher,
- defended contamination/ASR lower,
- some latency overhead introduced by defense stages.

---

## 9. Troubleshooting

### Local model missing

Symptom:

- local provider logs model path not found.

Resolution:

- run setup without `--no-local`, or
- set valid `LLAMA_MODEL_PATH`, or
- use `--infer ollama` / `--infer deepseek`.

### Want hard failure instead of fallback

- add `--no-local-fallback` to `src/main.py` or `src/experiment_runner.py`.

### ONNX runtime device warning

- non-fatal in many container environments; does not block execution.

---

## 10. Security and ethics

This repository is for defensive security research and robustness evaluation. Only test systems you own or are explicitly authorized to test.

---

## 11. Current status

Implemented and runnable:

- secure RAG CLI pipeline,
- setup verification tooling,
- poisoning simulation framework,
- defense orchestrator,
- benchmark runner with export artifacts.

Recommended first validation command:

```bash
python3 test_setup.py --no-local
```
