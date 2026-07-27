# 🏥 RAG Q&A Assistant on India’s Health Transformation

An end-to-end Retrieval-Augmented Generation (RAG) system built to answer questions about India’s health sector transformation strictly grounded in the official Press Information Bureau (PIB) backgrounder document.

---

## 📌 Project Architecture & Flow

```text
[ PIB Web Page / HTML ]
          │
          ▼ (1. Ingestion & Cleaning)
[ Raw Clean Text ]
          │
          ▼ (2. Chunking: 200-500 words + overlap)
[ Text Chunks ] ────► [ Embedding Model: all-MiniLM-L6-v2 ]
                               │
                               ▼
                    [ ChromaDB Vector Store ]
                               │
[ User Query ] ──► [ Similarity Search (Cosine) ]
                               │
                               ▼
                    [ Top-K Context Chunks ]
                               │
                               ▼
                     [ Grounded RAG Prompt ]
                               │
                               ▼
                     [ Google Gemini LLM ]
                               │
                               ▼
                    [ Grounded Answer + Sources ]

🚀 Repository Structure
rag-health-pib/
├── data/
│   ├── cleaned_pib_text.txt   # Raw extracted   clean document text
│   ├── chunks.json            # Parsed 200-500 word text chunks
│   └── chroma_db/             # Local ChromaDB vector database index
├── src/
│   ├── __init__.py
│   ├── ingestion.py           # Web scraping & text chunking script
│   ├── vector_store.py        # Embedding generation & ChromaDB vector store
│   └── rag_chain.py           # Retrieval pipeline & Gemini LLM response generation
├── app.py                     # Interactive Streamlit Web Interface
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── Implementation_Note.md     # 1-2 page implementation decisions & research note
└── README.md                  # System setup & technical guide


⚙️ Technical Details
    1.Document Ingestion & Chunking (src/ingestion.py)
        Source: Official PIB backgrounder page (PRID=2269699).
        Preprocessing: HTML boilerplate (headers, footers, scripts, navigation) stripped using BeautifulSoup.
        Chunking Strategy: Uses RecursiveCharacterTextSplitter configured for target chunks of 200–500 words (~2000 characters) with a 50-word overlap (~300 characters) to preserve contextual boundaries across key policy topics (AB-PMJAY, ABDM, PM-ABHIM, Ayushman Arogya Mandirs).

    2.Embeddings & Vector Storage (src/vector_store.py)
        Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional dense vectors, runs 100% locally on CPU).
        Vector Store: ChromaDB configured with Cosine Similarity Distance.
        Storage Location: Saved locally on disk at data/chroma_db/.

    3.RAG Pipeline & Hallucination Guardrails (src/rag_chain.py)
        Retrieval: Converts query into vector embedding and retrieves top-k(default k=3) most relevant context chunks.
        LLM Engine: Google Gemini API (google-genai SDK).
        Hallucination Prevention:
            1.System prompt strictly restricts answers to provided context snippets.
            2.Directs the LLM to explicitly output "Based on the provided document,I cannot answer this         question." if the query falls outside the document's scope.
            3.Displays source context chunks directly in the UI for complete transparency and auditability.

💻 Setup and Run Instructions
1. Prerequisites
    Python 3.10 or higher
    Google Gemini API Key (Get an API key here)
2. Installation Steps
    1.Clone the repository:
        git clone https://github.com/vrdn-rana/rag-health-pib.git
        cd rag-health-pib
    2.Create and activate a virtual environment:
        python -m venv venv
        # On Windows:
        venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
    3.Install dependencies:
        pip install -r requirements.txt
    4.Set up Environment Variables:
        Create a .env file in the root folder:
        GEMINI_API_KEY=your_actual_gemini_api_key_here
3. Execution Pipeline
    Execute the system step-by-step:
    Step 1: Download & Chunk Document
            python src/ingestion.py
            Output: Saves cleaned text to data/cleaned_pib_text.txt and chunks to data/chunks.json.
    Step 2: Generate Vector Index
            python src/vector_store.py
            Output: Generates embeddings and builds vector store in data/chroma_db/.
    Step 3: Test RAG Pipeline via CLI
            python src/rag_chain.py
    Step 4: Launch Web Interface
            streamlit run app.py