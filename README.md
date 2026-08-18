<div align="center">
  <h1>🎙️ Voice-Enabled RAG Agent</h1>
  <p><strong>A high-performance, Voice-to-Text Retrieval-Augmented Generation system.</strong></p>
  <p>Built for the HH Goa 2026 Shortlisting Task 2.</p>
</div>

---

## 🌟 Overview

This project features a Python FastAPI backend orchestrating a state-of-the-art LLM pipeline, a dense FAISS vector database indexed with semantic chunks, and a sleek Next.js glassmorphism frontend dashboard. 

It fulfills all technical and latency requirements by bridging real-time speech transcription with heavily-guardrailed, grounded knowledge retrieval.

## ✨ Key Features & Hackathon Requirements

### 🗣️ 1. Speech-to-Text Pipeline
Captures microphone audio in the browser and transcribes it via the **Sarvam AI STT API** (`saaras:v3`).

### 🧩 2. Advanced "Vast" Chunking Strategy
We didn't just use a naive fixed-size chunker. Our preprocessing pipeline (`backend/scripts/preprocess.py`) implements a multi-pronged approach to ensure high-quality retrieval:
- **Fixed-Size Chunking:** With optimal overlap to preserve boundary context.
- **Semantic Window Chunking:** Uses NLTK to split text by natural sentence boundaries rather than arbitrary character counts.
- **Metadata-Aware Chunking:** Prepends source queries directly into the chunk text so the LLM retains local context.

### 🛡️ 3. Robust Orchestration & Guardrails
Our orchestration layer (`backend/app/services/orchestrator.py`) handles the pipeline securely:
- **Off-topic Validation:** Rejects queries unrelated to the knowledge base before wasting compute.
- **Retrieval Thresholding:** Drops retrieved context if the similarity score is too low.
- **Anti-Hallucination Checking:** A secondary LLM pass strictly verifies that the final generated answer is 100% grounded in the retrieved facts.

### 🌍 4. Multilingual Support
Seamlessly handles queries in English, Hindi, and Bengali. Non-English queries are auto-translated for retrieval against English embeddings, and the final response is generated natively in the user's original language.

## 💻 Tech Stack
- **Frontend:** Next.js 14, React, Tailwind CSS
- **Backend:** FastAPI, Python
- **Vector DB:** FAISS
- **Embeddings:** `BAAI/bge-small-en-v1.5` (via `fastembed` ONNX runtime for blazing fast local CPU inference)
- **LLM Engine:** `groq/compound-mini` via Groq API
- **Speech-to-Text:** Sarvam AI STT API

## 🚀 Quickstart

### 1. Backend Setup
```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file inside `backend/` and add your keys:
```env
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
```

### 3. Build the Vector Index
Generate the chunks and build the FAISS index (Note: CPU embedding may take some time depending on record size).
```powershell
python scripts\preprocess.py --num_records 5000
python scripts\embed.py
```

### 4. Run the Application

**Start the Backend Server:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Start the Frontend Dashboard:**
```powershell
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to access the Voice-Enabled RAG Dashboard.

## ⚡ Benchmarks & Latency

We rigorously stress-tested the backend `/api/query` orchestration endpoint using a batch of randomized sequential queries. The crucial **Retrieval Latency** is safely under the **50ms** target required by the hackathon specifications.

| Metric | Retrieval Latency | Generation Latency | Total API Roundtrip |
|--------|-------------------|--------------------|---------------------|
| **P50 (Median)** | **21.37 ms** | 1241.80 ms | 6501.58 ms |
| **P70** | **22.41 ms** | 2372.20 ms | 10799.69 ms |
| **P100 (Max)** | **28.91 ms** | 2786.47 ms | 11528.15 ms |

*(Note: The total API roundtrip includes network transfer overhead, Groq API response times, and multi-step LLM guardrail evaluations.)*
