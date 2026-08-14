# Voice-Enabled RAG System (HH Goa 2026)

A high-performance Voice-Enabled Retrieval-Augmented Generation (RAG) system for HH Goa 2026 Task 2. This project features a Python FastAPI backend orchestrating a state-of-the-art LLM pipeline, a dense FAISS vector database indexed with over 53,000 semantic chunks, and a sleek Next.js glassmorphism frontend dashboard.

## Key Features
- **Voice-to-Text Pipeline:** Captures microphone audio in the browser, transcribes it via the Sarvam STT API (saaras:v3), and processes the semantic query.
- **Robust Orchestration Guardrails:**
  - **Off-topic Validation:** Rejects queries unrelated to the knowledge base.
  - **Retrieval Thresholding:** Rejects questions if retrieved context scores fall below 0.60.
  - **Hallucination Checking:** A secondary LLM agent strictly verifies that the final generated answer is strictly grounded in the retrieved facts.
- **Multilingual Support:** Seamlessly handles queries in English, Hindi, and Bengali. Non-English queries are auto-translated for retrieval against English embeddings, and the final response is generated natively in the user's original language.
- **Massive Scalability:** The local FAISS database is indexed with 53,285 semantic chunks from the MSMARCO-XI dataset, with vector search returning results in under 50 milliseconds.

## Tech Stack
- **Frontend:** Next.js 14, React, Tailwind CSS
- **Backend:** FastAPI, Python
- **Vector DB:** FAISS
- **Embeddings:** `BAAI/bge-small-en-v1.5` (via ONNX runtime for blazing fast local CPU inference)
- **LLM Engine:** `llama-3.1-8b-instant` via Groq API
- **Speech-to-Text:** Sarvam STT API

## Quickstart

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
Generate the 53K+ chunks and build the FAISS index (Note: CPU embedding may take ~15 minutes).
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

## Benchmarks & Latency

We rigorously stress-tested the backend `/api/query` orchestration endpoint using a batch of 50 randomized sequential queries against the 53,285-chunk database.

| Metric | Retrieval Latency | Generation Latency | Total API Roundtrip |
|--------|-------------------|--------------------|---------------------|
| **P50 (Median)** | 15.21 ms | 75.15 ms* | 2.55s |
| **P70** | 17.07 ms | 2.29s | 9.48s |
| **P100 (Max)** | 50.27 ms | 2.44s | 9.92s |

*(Note: The lightning-fast 75ms P50 generation time includes fail-fast rejection for queries caught by the off-topic guardrails.)*
