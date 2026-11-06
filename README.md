# 📚 AI Study Assistant: Production-Grade RAG Engine

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

## 🎯 Why This Qualifies as Genuine RAG

| Criteria | Naive Prompt Stuffing | Fake / Superficial RAG | Genuine RAG (Our Implementation) |
| :--- | :--- | :--- | :--- |
| **Document Processing** | Concatenates whole document into prompt | Claims RAG but passes full text | Page extraction, cleaning, boundary chunking |
| **Vector Representation** | None | Keyword search / Regex matching | Dense embeddings (all-MiniLM-L6-v2, 384-dim) |
| **Vector Store** | None | In-memory list of strings | `faiss.IndexFlatIP` with L2 normalization |
| **Semantic Retrieval** | None | Arbitrary text slicing | Dense query vector search for Top-$k$ nearest neighbors |
| **Relevance Filtering** | None | Blindly sends Top-$k$ | Configurable cosine threshold with short-circuiting |
| **Context Augmentation** | Stuffs whole document | Stuffs raw unranked chunks | Formats verified retrieved chunks with page provenance |
| **LLM Generation** | Unconstrained generation | Unconstrained generation | Grounded generation constrained strictly to retrieved context |
| **Source Provenance** | Model invents citations | Model hallucinates pages | Python extracts exact page numbers from chunk metadata |

---

## 🎤 15–20 Difficult RAG Interview Questions & Strong Answers

### Q1: Why do we need chunking? Why not embed individual sentences or whole pages?
> **Answer**: Embedding individual sentences strips away essential co-reference context (e.g. pronouns like "this algorithm" or "it" lose their referents) and leads to index bloat. Embedding whole pages averages hundreds of distinct thoughts together into a single 384-dimensional vector, causing semantic dilution where specific factual queries fail to achieve high similarity. Chunking at 800–1000 characters preserves cohesive semantic units while maintaining high retrieval precision.

### Q2: What mathematically happens during L2 normalization and why does it make Inner Product equal to Cosine Similarity?
> **Answer**: L2 normalization scales vector $\mathbf{v}$ by its Euclidean magnitude $\sqrt{\sum v_i^2}$, ensuring $\|\mathbf{v}\|_2 = 1.0$. The formula for cosine similarity is $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$. When $\|\mathbf{u}\|_2 = 1.0$ and $\|\mathbf{v}\|_2 = 1.0$, the denominator is $1.0$, leaving $\mathbf{u} \cdot \mathbf{v} = \cos(\theta)$. Thus, `faiss.IndexFlatIP` directly computes exact cosine similarity without square-root operations during search.

### Q3: Why did you choose `faiss.IndexFlatIP` instead of `IndexIVFFlat` or `HNSW`?
> **Answer**: `IndexFlatIP` is an exact, brute-force inner product index. For personal study documents (under 2,000 chunks), computing 2,000 dot products of dimension 384 takes less than 1 millisecond on a standard CPU. Approximate Nearest Neighbor (ANN) algorithms like IVFFlat (inverted file clustering) or HNSW (hierarchical navigable small world graphs) trade off recall accuracy for sub-linear search speed on million-vector datasets. Using ANN here would introduce training overhead and false negatives with zero perceptible latency benefit.

### Q4: Why can't we rely on Top-$k$ retrieval alone? Why is a similarity threshold required?
> **Answer**: Top-$k$ retrieval is an unconditional nearest neighbor query: it *always* returns $k$ vectors regardless of how distant they are in semantic space. If a user asks a question completely absent from the document (e.g., asking for baking instructions in a legal contract), Top-$k$ will still return the $k$ "least distant" chunks. Passing these irrelevant chunks to the LLM invites hallucination. A similarity threshold acts as an admission gatekeeper, enabling short-circuiting before LLM invocation.

### Q5: Is a similarity threshold of 0.35 universally optimal? How should it be tuned?
> **Answer**: No threshold is universally optimal. Cosine similarity distributions depend heavily on the embedding model, the domain vocabulary, and the chunk length. A threshold should be determined empirically using a validation dataset of in-domain queries (which should pass) and out-of-domain/adversarial queries (which should fail), maximizing the F1 score between precision and recall of the retrieval gate.

### Q6: Why must the exact same embedding model be used for both documents and user queries?
> **Answer**: Dense embeddings map text into a learned coordinate manifold where semantic similarity corresponds to geometric proximity. Different embedding models (or different parameter checkpoints) produce completely different vector spaces with different dimensions, orientations, and latent coordinate representations. Comparing a query embedded by Model A against chunks embedded by Model B in FAISS is mathematically meaningless noise.

### Q7: What is the difference between bi-encoders and cross-encoders in RAG?
> **Answer**: Bi-encoders (like `all-MiniLM-L6-v2`) embed query and document independently into fixed-size dense vectors. This allows offline pre-computation and sub-millisecond vector index lookup, but limits interaction to late dot-product pooling. Cross-encoders pass both query and document simultaneously through all transformer self-attention layers, computing full token-to-token cross-attention. Cross-encoders achieve significantly higher ranking precision but are computationally expensive ($O(N)$ transformer forward passes per query), making them ideal for reranking a small candidate pool rather than primary vector search.

### Q8: What is chunk deduplication and why is it necessary if we already have FAISS?
> **Answer**: Because chunking uses a sliding window with overlap (e.g. 180 characters), consecutive chunks often contain substantially duplicate sentences. If both chunks score highly in FAISS, passing both to the LLM consumes context window tokens redundantly without adding new information. Our lightweight Jaccard token overlap filter suppresses chunks with $>60\%$ token overlap with higher-ranking chunks, maximizing information density in the prompt.

### Q9: How is source attribution implemented, and why shouldn't the LLM generate page citations freely?
> **Answer**: LLMs are probabilistic token predictors prone to confabulation; if asked to cite page numbers, they may generate plausible-looking but non-existent citations. In our system, source attribution is deterministic: Python tracks the `page_number` metadata on every retrieved `DocumentChunk` object that passes the similarity threshold. The UI displays `Sources: Page X, Page Y` directly from metadata, independent of the LLM's text output.

### Q10: What happens if a user uploads a scanned PDF image with no OCR text layer?
> **Answer**: `pdfplumber` relies on the PDF's internal text stream. If a PDF consists of scanned raster images with no embedded text, `extract_text()` returns empty strings. Our `text_processor` checks for empty extraction and halts ingestion early, presenting a clear diagnostic error: *"No readable text found. The document may be a scanned image lacking an OCR layer."*

### Q11: What is the "Lost in the Middle" phenomenon and how does RAG mitigate it?
> **Answer**: Research (Liu et al., 2023) demonstrated that LLMs attend most effectively to information placed at the very beginning and very end of their context window, while information in the middle suffers significant retrieval degradation. RAG mitigates this by extracting only the top 3–5 highly relevant chunks, avoiding monolithic context stuffing and keeping the prompt compact and focused.

### Q12: Why shouldn't full-document summarization be performed using standard RAG retrieval?
> **Answer**: RAG is a *needle-in-a-haystack* query-directed mechanism: it retrieves a localized cluster of chunks that match a specific question. Summarization is a *global synthesis* task requiring uniform coverage across the entire document. Running top-$k$ retrieval for a summary would sample a narrow topical pocket and ignore the rest of the document. For large documents, the correct approach is hierarchical map-reduce summarization.

### Q13: What is the time and space complexity of our FAISS search?
> **Answer**: With $N$ chunks and embedding dimension $d=384$:
> - Space complexity: $O(N \cdot d)$ float32 values (for 1,000 chunks $\approx 1.5$ MB).
> - Time complexity: $O(N \cdot d)$ floating-point operations per query. For $N=1,000$, this is $\approx 384,000$ FLOPs, executing in under 0.5 milliseconds on modern CPU SIMD/AVX registers.

### Q14: How does our RAG pipeline prevent hallucination when a question is unanswerable from the document?
> **Answer**: Two-layer defense:
> 1. **Retrieval Layer**: If cosine similarity is below `SIMILARITY_THRESHOLD`, the pipeline short-circuits immediately. Gemini is never called.
> 2. **Prompt Layer**: If borderline chunks pass, the system prompt strictly instructs the model to rely only on provided excerpts, forbids extrapolating from external knowledge, and mandates returning a standardized disclaimer if facts are absent.

### Q15: What is Recall@k and how is it calculated in our evaluation script?
> **Answer**: $\text{Recall@}k$ (also called $\text{Hits@}k$) measures whether the ground-truth relevant document chunk or page is present among the top-$k$ retrieved candidates:
> $$\text{Recall@}k = \frac{\sum_{i=1}^{Q} \mathbb{I}(\text{target} \in \text{Top-}k(Q_i))}{Q}$$
> Our `eval_rag.py` computes Recall@1, Recall@3, and Recall@5 against curated test queries with known ground truth source pages.

### Q16: What are the limitations of this implementation?
> **Answer**:
> 1. **Pure Dense Retrieval**: Dense semantic search can struggle with exact keyword matching, specialized serial numbers, or rare terminology where lexical search (BM25) excels.
> 2. **Fixed Chunk Boundaries**: Static character-based windowing does not adapt dynamically to varying document hierarchy (headers, sub-sections, tables).
> 3. **Single-Turn Q&A**: Each question is retrieved independently without conversational query rewriting or multi-turn history resolution.
> 4. **No Scanned OCR Support**: Scanned PDFs without embedded text streams require an upstream OCR engine (e.g. Tesseract).

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
