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
