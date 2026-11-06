# 📚 AI Study Assistant: A Modular RAG Application

A technically defensible, lightweight Retrieval-Augmented Generation (RAG) study assistant built with **Python**, **pdfplumber**, **Sentence Transformers**, **FAISS**, **Gemini API**, and **Gradio**.

This application transforms the naive document-stuffing approach into a genuine two-stage RAG pipeline:
1. **Offline Ingestion & Vector Indexing**: Document extraction, artifact cleaning, boundary-aware chunking, dense vector embeddings, and local FAISS indexing.
2. **Online Query-Time RAG**: Semantic query embedding, inner product similarity search, chunk deduplication, relevance thresholding, grounded generation, and deterministic source attribution.

---

## 🏗️ System Architecture: Offline vs. Online Pipeline

```
========================================================================================
STAGE 1: OFFLINE INGESTION & INDEXING PIPELINE (Document Ingestion)
========================================================================================

   Uploaded PDF
        │
        ▼
   [pdfplumber] ─────────► Extract raw text per page with 1-indexed page numbers
        │
        ▼
   [Text Cleaning] ──────► Clean excessive whitespace, non-printable chars, broken hyphens
        │
        ▼
   [Recursive Chunker] ──► Split at paragraphs -> sentences -> words
        │                  Configurable: CHUNK_SIZE=900, CHUNK_OVERLAP=180
        │                  Metadata attached: {chunk_id, page_number, text, start_char, end_char}
        │
        ▼
   [Embedding Model] ────► sentence-transformers/all-MiniLM-L6-v2 (384-dim)
        │                  L2 Normalization applied: ||v||_2 = 1.0
        │
        ▼
   [FAISS Vector Store] ─► faiss.IndexFlatIP(384) + In-memory Chunk Metadata Store
                           (Inner Product of unit vectors == Cosine Similarity)

========================================================================================
STAGE 2: ONLINE QUERY-TIME RAG PIPELINE (Question Answering)
========================================================================================

   User Question
        │
        ▼
   [Query Embedding] ────► all-MiniLM-L6-v2 (L2 normalized: ||q||_2 = 1.0)
        │
        ▼
   [FAISS Top-k Search] ─► Exhaustive inner-product search (returns k candidate chunks + scores)
        │
        ▼
   [Deduplication] ──────► Remove redundant/highly overlapping chunks (Jaccard token overlap)
        │
        ▼
   [Relevance Filter] ───► Filter by SIMILARITY_THRESHOLD (default: 0.35)
        │
   ┌────┴──────────────────────────┐
   │ No chunks pass threshold?     │ Any chunks pass?
   ▼                               ▼
[Short-Circuit Exit]        [Context Assembly] ──► Format text excerpts with [Page X] tags
"I couldn't find enough...        │
(No LLM call!)                    ▼
                            [Grounded Prompt]  ──► Strict instruction: answer ONLY from context,
                                  │                do not invent facts or general knowledge
                                  ▼
                            [Gemini LLM]       ──► Grounded generation (temp=0.2)
                                  │
                                  ▼
                            [Output Assembly]  ──► Grounded Answer
                                                   + Deterministic Sources (from metadata)
                                                   + Retrieval Debug Inspection
```

---

## 📁 Project Structure

```
AI-study/
├── rag/                                # Core Modular RAG Engine
│   ├── __init__.py                     # Package exports
│   ├── config.py                       # Tunable constants: chunk size, overlap, threshold, top-k
│   ├── text_processor.py               # pdfplumber page extraction & artifact cleaning
│   ├── chunker.py                      # Boundary-aware recursive chunker with metadata tracking
│   ├── embedder.py                     # SentenceTransformer wrapper + L2 normalization
│   ├── vector_store.py                 # FAISS IndexFlatIP & chunk metadata registry
│   ├── deduplicator.py                 # Lightweight lexical/token overlap deduplication
│   └── pipeline.py                     # High-level coordinator (Ingestion & Retrieval)
├── gemini_api.py                       # Grounded Gemini prompt, map-reduce summary, quiz generation
├── pdf_reader.py                       # Backward-compatible adapter delegating to rag.text_processor
├── app.py                              # Modern Gradio UI with Debug Inspector & Source Badges
├── test_rag.py                         # Automated test suite (unit + integration tests)
├── eval_rag.py                         # Standalone Recall@k retrieval evaluation benchmark
├── requirements.txt                    # Project dependencies
└── README.md                           # Comprehensive documentation & interview defense guide
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux / macOS / Colab:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. API Key Configuration

Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Running the Application

```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 4. Running in Google Colab

To run this project inside Google Colab:
1. Upload the project folder or clone your GitHub repository into the Colab runtime.
2. In a notebook cell, install the dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Set your environment variable:
   ```python
   import os
   os.environ["GEMINI_API_KEY"] = "your_key_here"
   ```
4. Launch Gradio with public sharing enabled:
   ```python
   import app
   app.demo.launch(share=True)
   ```

---

## 🧪 Verification & Evaluation

### Automated Test Suite
Run the unit test suite covering chunking, normalization, cosine similarity math, FAISS search, deduplication, and threshold short-circuiting:

```bash
python test_rag.py
```

### Retrieval Evaluation Benchmark (Recall@k)
Evaluate semantic retrieval accuracy on a multi-topic benchmark corpus:

```bash
python eval_rag.py
```
**Output Metrics:**
- **Recall@1**: Percentage of queries where the ground-truth page is retrieved at Rank 1.
- **Recall@3**: Percentage of queries where the ground-truth page is within the Top 3 results.
- **Recall@5**: Percentage of queries where the ground-truth page is within the Top 5 results.

---

## 🧠 Deep-Dive Technical Explanation

### 1. Chunking Strategy & Natural Boundaries
- **Chunk Size (800–1000 characters / ~150–180 words)**:
  - If chunks are too small (e.g., 100 characters), they lack semantic context, causing fragmented embeddings that cannot answer nuanced questions.
  - If chunks are too large (e.g., 4000 characters), multiple disparate concepts get averaged together in the embedding vector, diluting the similarity score for specific factual queries.
  - 800–1000 characters provides the sweet spot for academic and study materials: enough room for a complete premise, mechanism, and conclusion.
- **Overlap (150–200 characters / ~20%)**:
  - Overlap prevents boundary clipping: if an essential fact or entity relation occurs across chunk boundaries, overlap ensures that at least one chunk captures the complete thought intact.
- **Boundary-Aware Splitting**:
  - Instead of blind character slicing, our chunker recursively searches backwards for natural linguistic boundaries: paragraph breaks (`\n\n`), then sentence terminators (`. `, `? `, `! `), then words. Words and sentences are never severed mid-stream.
- **Metadata Provenance**:
  - Every `DocumentChunk` object stores: `chunk_id`, `page_number`, `start_char`, `end_char`, and `text`.

### 2. Embeddings & Mathematical Equivalence
- **Embedding Model (`sentence-transformers/all-MiniLM-L6-v2`)**:
  - A lightweight and widely used baseline that provides a good trade-off between semantic quality, embedding dimensionality (384), CPU inference speed, and minimal resource consumption (~80MB footprint).
  - The exact same model weights must be used to embed both document chunks and user queries so that both reside in the exact same shared semantic vector space.
- **L2 Normalization**:
  - Every embedding vector $\mathbf{v}$ is divided by its Euclidean norm:
    $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{d} v_i^2}}$$
- **Inner Product = Cosine Similarity**:
  - The cosine similarity between two vectors $\mathbf{u}$ and $\mathbf{v}$ is:
    $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  - When both vectors are pre-normalized such that $\|\mathbf{u}\|_2 = 1$ and $\|\mathbf{v}\|_2 = 1$:
    $$\mathbf{u}_{\text{norm}} \cdot \mathbf{v}_{\text{norm}} = \cos(\theta)$$
  - This allows FAISS `IndexFlatIP` (which computes fast dot products) to compute exact Cosine Similarity with zero mathematical approximation.

### 3. FAISS Vector Indexing
- **Index Type (`IndexFlatIP`)**:
  - An exact, brute-force flat index computing inner products across all vectors in memory.
  - For typical study documents (1 to 500 pages, generating 50 to 2,000 chunks), exhaustive flat search takes less than **1 millisecond** on CPU and achieves **100% recall** (zero false negatives).
  - Approximate Nearest Neighbor (ANN) indexes (such as `IndexIVFFlat` or `HNSW`) require clustering training steps, increase index build latency, and trade away recall accuracy for speed—unnecessary for documents of this scale.
- **Local In-Memory vs. Hosted Vector DB**:
  - FAISS runs entirely locally in RAM, requires no external infrastructure, costs $0, has no network latency, and protects user document privacy.

### 4. Retrieval Deduplication & Relevance Thresholding
- **Lightweight Deduplication**:
  - Consecutive overlapping chunks often contain redundant sentences. The pipeline calculates pairwise token Jaccard similarity across candidate chunks and discards lower-ranking chunks with $>60\%$ token overlap, maximizing context diversity.
- **Relevance Threshold & Short-Circuiting**:
  - Top-$k$ search by itself will *always* return $k$ chunks, even if the user asks a completely unrelated question (e.g. asking about cooking recipes in a physics document).
  - By applying a configurable similarity cutoff (`SIMILARITY_THRESHOLD = 0.35`), chunks scoring below the cutoff are discarded.
  - **Short-circuiting**: If no chunks meet the threshold, the system halts immediately and returns:
    > *"I couldn't find enough relevant information in the uploaded document to answer this question."*
  - The Gemini API is **never called**, completely preventing hallucination and saving API costs.

### 5. Grounded Generation & Deterministic Source Attribution
- **Anti-Hallucination Grounded Prompt**:
  - The prompt explicitly instructs Gemini to act as a strict factual assistant, use *only* the retrieved excerpts, reject extrapolation, and acknowledge when information is missing. Temperature is locked to `0.2` for deterministic output.
- **Deterministic Source Attribution**:
  - Gemini is **never** trusted to invent or guess page numbers.
  - The application deterministically extracts the unique set of page numbers directly from the `DocumentChunk.page_number` metadata of the verified retrieved chunks, displaying them as: `📌 Sources: Page X, Page Y`.

### 6. Global Document Synthesis (Summarization & Quiz)
- **Why Summarization & Quiz are Separate from Query RAG**:
  - Query-based RAG retrieves a small neighborhood of chunks relevant to a specific query.
  - Full-document summarization is a **global synthesis task**. Applying Top-$k$ retrieval to a summary would discard 80–90% of the document content, resulting in an incomplete and biased summary.
  - Therefore, summarization operates on the full document text. For large documents exceeding `MAX_SUMMARY_CHUNK_CHARS` (25,000 chars), the system applies a **hierarchical map-reduce summarization** strategy (summarizing sections first, then synthesizing the executive summary).

---

## 🚀 Future Work & Production Roadmap

1. **Hybrid Search (Dense + Sparse)**:
   - Combine FAISS dense semantic search with BM25 keyword search using Reciprocal Rank Fusion (RRF) to excel at both conceptual queries and exact keyword/acronym lookups.
2. **Cross-Encoder Reranking**:
   - Add a lightweight secondary reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to rescore the top-20 FAISS candidates before selecting the top-4 for prompt injection.
3. **Agentic / Semantic Chunking**:
   - Split chunks based on semantic embedding distance between consecutive sentences or document layout tags (headers, Markdown trees) rather than static character lengths.
4. **Conversational Memory & Query Rewriting**:
   - Use a lightweight LLM call to rewrite multi-turn follow-up questions (e.g., "What were its key advantages?") into standalone search queries before embedding.
5. **RAG Triad Automated Evaluation (Ragas / TruLens)**:
   - Formally measure the three pillars of RAG: **Context Precision** (signal-to-noise of retrieval), **Groundedness / Faithfulness** (absence of hallucination), and **Answer Relevance** (directness of response).
6. **Vector Database Migration**:
   - For multi-tenant production scaling: replace in-memory FAISS with a persistent, distributed vector database such as Qdrant, Weaviate, or pgvector with metadata filtering (by user ID, document ID, upload date).

---

## 📜 License & Acknowledgments

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.
- Built with [Google Gemini API](https://ai.google.dev/), [Hugging Face Sentence Transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss), [Gradio](https://www.gradio.app/), and [pdfplumber](https://github.com/jsvine/pdfplumber).
