# RAG Poisoning Project

Research/engineering project exploring **poisoning attacks against Retrieval-Augmented Generation (RAG)** systems and practical defenses. The focus is on how adversarial or low-quality content can enter a RAG knowledge source (documents, vector DB, web crawl, user uploads) and subsequently **influence retrieved context** and model outputs.

> Status: WIP / experimental (expect breaking changes).

---

## What is “RAG poisoning”?

In a typical RAG pipeline, a system:
1. collects documents,
2. chunks them,
3. embeds chunks into a vector store,
4. retrieves top-k chunks for a query, and
5. prompts an LLM with the retrieved context.

**Poisoning** happens when an attacker (or accidental data quality issue) causes harmful / misleading / instruction-like content to be indexed so that it is retrieved and used during generation. This can lead to:
- misinformation or targeted bias,
- prompt-injection style instructions inside retrieved text (“ignore previous instructions…”),
- data exfiltration attempts (“print secrets”), and
- degraded answer quality.

---

## Goals

- Implement a baseline RAG pipeline suitable for controlled experiments.
- Simulate and measure common poisoning strategies (e.g., injected passages, keyword traps, embedding-space manipulation, instruction injection).
- Provide evaluation harnesses to quantify:
  - retrieval contamination rate,
  - attack success rate (ASR),
  - factuality/faithfulness degradation,
  - robustness under filtering/defenses.
- Experiment with defenses (sanitization, ranking-time filters, provenance scoring, chunk-level policies, etc.).

---

## Threat model (assumptions)

This project generally assumes:
- The attacker can introduce content into one of the system’s sources (direct upload, public wiki/web, shared drive, etc.).
- The model weights are not modified; the attack targets **retrieval + context**.
- Success is defined as the model producing attacker-desired behavior/content **because poisoned chunks were retrieved**.

Adjust these assumptions to match your deployment context.

---

## Key ideas & experiments (examples)

You can adapt these to whatever you’re implementing in this repo:

### 1) Prompt-injection via retrieved text
Poisoned chunk contains instruction-like content that tries to override system rules.

### 2) Keyword/SEO “trap” poisoning
Documents are crafted to match many queries and dominate retrieval.

### 3) Targeted misinformation
Poisoned content is inserted for specific entities/topics and only triggers for certain queries.

### 4) Embedding manipulation (conceptual)
Adversarial text crafted to land near many query embeddings (implementation depends on embedding model).

---

## Getting started

### Prerequisites
- Python 3.10+ (recommended)
- An LLM provider (OpenAI / local model) depending on your setup
- An embeddings model (provider or local)
- A vector store (FAISS / Chroma / Pinecone / etc.) depending on your setup

### Setup

```bash
# (Optional) create venv
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# install deps (adjust to your repo)
pip install -r requirements.txt
```

If your repository uses Poetry or uv instead, replace the above accordingly.

### Configure environment

Create a `.env` (or export env vars) as needed, e.g.:

```bash
# Example only — adjust to your stack
OPENAI_API_KEY=...
EMBEDDING_MODEL=...
LLM_MODEL=...
VECTOR_DB_PATH=./data/vector_store
```

---

## How to run (fill in with actual commands)

Because repo layouts differ, here are common entry points. Replace with the real scripts/modules in this project:

### 1) Ingest documents
```bash
python -m rag_poisoning.ingest --input ./data/docs --out ./data/index
```

### 2) Build / update vector index
```bash
python -m rag_poisoning.index --index ./data/index
```

### 3) Query the RAG system
```bash
python -m rag_poisoning.query --question "..." --top_k 5
```

### 4) Run poisoning experiment suite
```bash
python -m rag_poisoning.experiments.run --suite baseline
```

### 5) Evaluate
```bash
python -m rag_poisoning.eval --run_id <ID>
```

---

## Evaluation

Suggested metrics (use whichever are implemented here):
- **Retrieval contamination rate**: fraction of retrieved chunks that are poisoned.
- **Attack Success Rate (ASR)**: fraction of queries where output matches attacker goal.
- **Faithfulness / groundedness**: whether output is supported by safe context.
- **Robustness**: performance under defenses (filters/rerankers/policies).

---

## Defenses to explore

Potential defenses you might implement/compare:
- **Chunk sanitization**: strip or down-rank instruction-like patterns.
- **Provenance & allowlists**: only retrieve from trusted sources or signed documents.
- **Reranking with safety criteria**: use a reranker that penalizes injection patterns.
- **Context firewall**: classify retrieved text; block or mask unsafe segments.
- **Prompt hardening**: explicitly instruct model to treat retrieved text as untrusted.
- **Diversity constraints**: avoid dominance of a single source/domain.
- **Continuous monitoring**: detect drift in retrieval distribution.

---

## Project structure (update to match repo)

Example layout:

```text
.
├─ data/
├─ notebooks/
├─ src/ or rag_poisoning/
├─ tests/
├─ requirements.txt / pyproject.toml
└─ README.md
```

---

## Safety & ethics

This repository is intended for **defensive research and robustness testing**. Do not deploy poisoning techniques against systems you do not own or have explicit permission to test.

If you publish results, clearly describe the threat model and limitations.

---

## Contributing

- Issues and PRs welcome.
- Please include reproducible steps and logs for bugs.
- Add tests for new features where possible.

---

## License

Add a license file (e.g., MIT/Apache-2.0) and update this section.

---

## Citation

If you use this work in academic or public research, consider adding a `CITATION.cff` or BibTeX entry here.
