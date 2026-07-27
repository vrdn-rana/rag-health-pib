# 📄 Implementation & Architectural Choices Note

**Project:** Grounded RAG Assistant on India’s Health Transformation  
**Data Source:** PIB Backgrounder (`https://www.pib.gov.in/PressReleasePage.aspx?PRID=2269699`)  
**Timebox:** 6–8 Hours  

---

## 1. Architectural Choices & Justifications

### A. Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
* **Why chosen:** 
  1. **Local & Cost-Effective:** Generates high-quality 384-dimensional dense vectors locally on CPU without requiring paid external embedding API calls.
  2. **Speed & Performance:** Extremely fast inference speed (~15ms per chunk) while maintaining strong semantic matching performance on English benchmark datasets (MTEB).
  3. **Zero Latency Overhead:** Eliminates network overhead during vector index construction and retrieval.

### B. Storage & Vector Index: `ChromaDB`
* **Why chosen:**
  1. **Embedded & Persistent:** ChromaDB runs in-process and persists vectors directly to disk (`data/chroma_db`) without requiring Docker containers or cloud vector database instances.
  2. **HNSW Cosine Indexing:** Built-in support for Hierarchical Navigable Small World (HNSW) indexing using cosine distance, allowing instant top-$k$ nearest neighbor search.
  3. **Metadata Storage:** Stores raw text chunks alongside metadata (`chunk_id`, `title`, `word_count`), enabling context display in the UI without secondary database lookups.

### C. LLM Engine & Prompt Design: Google Gemini (`gemini-flash` series)
* **Model Selection:** Google Gemini Flash was selected for its fast generation latency, strong instruction-following capabilities, and cost efficiency.
* **Prompt Engineering Strategy:**
  To guarantee factual grounding and zero hallucination, the prompt incorporates explicit negative constraints:
  - **Strict Boundaries:** Instructs the LLM to answer using *ONLY* provided context snippets.
  - **Fallback Instruction:** Directs the model to output: *"Based on the provided document, I cannot answer this question"* whenever information is absent from the context.
  - **Context Structure:** Context chunks are clearly delimited with headers and titles, giving the model structural clarity over source fragments.

---

## 2. Key Learnings & Research Insights

1. **Optimal Chunk Size for Government Policy Documents:**
   - Government releases contain dense, statistical facts (*e.g., scheme coverage figures, financial allocations*).
   - Smaller chunks (<150 words) frequently fragment tables and key context, while oversized chunks (>700 words) dilute embedding specificity.
   - Configuring chunk sizes to **200–500 words (~350 words average)** with a 50-word overlap maintained topic coherence (*e.g., keeping AB-PMJAY statistics together*).

2. **Dynamic LLM API Model Discovery:**
   - Google AI Studio updates API model endpoints dynamically. To make the pipeline resilient against model deprecations, a dynamic discovery function (`client.models.list()`) was integrated to automatically select active Gemini models assigned to the API key.

3. **Grounding Verification:**
   - Implementing transparent source attribution in the Streamlit interface (*showing retrieved chunks alongside answers*) provided immediate auditability, proving every generated figure (*e.g., 44.14 crore Ayushman cards*) was sourced directly from the document.

---

## 3. System Limitations & 2-Day Improvement Roadmap

If allocated **2 additional days**, the following upgrades would be implemented:

1. **Hybrid Retrieval (Dense + Sparse Search):**
   * *Limitation:* Pure semantic search occasionally struggles with exact key-term queries (*e.g., specific scheme acronyms or exact numerical figures*).
   * *Improvement:* Implement **Hybrid Search** combining ChromaDB dense vector retrieval with **BM25 keyword search**, re-ranked using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).

2. **Parent-Child / Hierarchical Chunking:**
   * *Limitation:* Fixed chunking can separate sub-headings from their parent policy background.
   * *Improvement:* Implement parent-child indexing—searching small child chunks (100 words) for high semantic similarity, but passing the larger parent section (500 words) as context to the LLM.

3. **Automated RAG Evaluation Framework (Ragas):**
   * *Limitation:* Verification was conducted manually via sample test queries and negative testing.
   * *Improvement:* Integrate **Ragas / G-Eval** framework to measure automated quantitative metrics:
     - **Faithfulness Score:** Verifies zero statements outside context snippets.
     - **Answer Relevance Score:** Measures query alignment.

4. **Multi-Format Document Parsing:**
   * *Improvement:* Extend the ingestion pipeline using `Unstructured` or `PyPDF` to dynamically handle PDF downloads and complex HTML tables natively.