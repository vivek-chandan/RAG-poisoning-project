# RAG Poisoning Proof of Concept

This repository demonstrates a Proof of Concept (PoC) for stealthy prompt injection and poisoning in Retrieval-Augmented Generation (RAG) systems through vector database embeddings.

It shows how malicious instructions embedded inside documents can influence the behavior of Large Language Models when those documents are retrieved during inference.

⚠️ This project is intended strictly for educational and security research purposes.

---

## Project Structure

```
├── src/            # Core source code
├── data/           # Vector database storage
├── models/         # Downloaded models (ignored by git)
├── logs/           # Runtime logs
├── setup.sh        # Environment setup script
├── test_setup.py   # Setup verification
├── requirements.txt
├── .env.example
├── .keys.example
```

---

## Setup

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
python3 test_setup.py
```

---

## Run the Demo

```bash
python3 src/rag_poisoning_demo.py
```

Remote inference:
```bash
python3 src/rag_poisoning_demo.py --infer ollama
python3 src/rag_poisoning_demo.py --infer deepseek
```

---

## Disclaimer

This PoC demonstrates security weaknesses in RAG pipelines.  
Do not use it for malicious purposes. Always follow ethical research practices.
