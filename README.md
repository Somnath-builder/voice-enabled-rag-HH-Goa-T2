# Voice-Enabled RAG System (HH Goa 2026)

A high-performance Voice-Enabled Retrieval-Augmented Generation (RAG) backend for HH Goa 2026 Task 2.

## Tech Stack
- **Vector DB:** FAISS
- **Embeddings:** `BAAI/bge-small-en-v1.5` (via ONNX runtime)
- **LLM:** `llama-3.1-8b-instant` via Groq
- **Framework:** FastAPI

## Quickstart

1. **Install Dependencies:**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Setup API Keys:**
   Create `.env` inside `backend/` and add your keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   SARVAM_API_KEY=your_sarvam_api_key
   ```

3. **Build the Vector Index:**
   ```powershell
   python scripts\preprocess.py
   python scripts\embed.py
   ```

4. **Run the Server:**
   ```powershell
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   Test the API at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
