"""Lightweight Retrieval Evaluation Script for RAG Pipeline.

Evaluates semantic retrieval accuracy using the standard information retrieval metric:
Recall@k (Hits@k) = Fraction of test queries where the ground-truth relevant page
                    appears within the top-k retrieved chunks.

Evaluates Recall@1, Recall@3, and Recall@5 on a multi-page benchmark corpus.
"""

from typing import List, Dict, Any
from rag.chunker import chunk_document
from rag.embedder import EmbeddingGenerator
from rag.vector_store import FAISSVectorStore
from rag.deduplicator import deduplicate_chunks

# Multi-page synthetic study corpus covering diverse topics across separate pages
BENCHMARK_PAGES = [
    (
        1,
        "Supervised Learning and Gradient Descent:\n"
        "In supervised machine learning, an algorithm learns a mapping function from input features to output targets "
        "using labeled training data. Linear regression models continuous targets by minimizing the Mean Squared Error (MSE). "
        "Gradient descent optimizes the model weights iteratively by computing the partial derivatives of the loss function "
        "with respect to each parameter. The learning rate hyperparameter controls the step size taken along the negative gradient direction."
    ),
    (
        2,
        "Convolutional Neural Networks (CNNs) for Computer Vision:\n"
        "Convolutional Neural Networks are designed to process grid-structured data such as images. "
        "A convolutional layer applies learnable filters or kernels across the spatial dimensions of an input tensor, "
        "producing feature activation maps that detect local edges, textures, and higher-order visual patterns. "
        "Pooling layers, such as max pooling, downsample spatial resolution to provide translation invariance and reduce parameters."
    ),
    (
        3,
        "Transformer Architecture and Scaled Dot-Product Self-Attention:\n"
        "The Transformer architecture replaces recurrent connections with multi-head self-attention mechanisms. "
        "Given input token embeddings, linear projections generate Query (Q), Key (K), and Value (V) matrices. "
        "Scaled dot-product attention computes compatibility between queries and keys as softmax((Q @ K.T) / sqrt(d_k)) @ V. "
        "This enables tokens to dynamically attend to all other tokens across the sequence in parallel, capturing long-range dependencies."
    ),
    (
        4,
        "Reinforcement Learning and Markov Decision Processes:\n"
        "Reinforcement learning formalizes decision-making problems using Markov Decision Processes (MDPs), defined by "
        "the tuple (S, A, P, R, gamma) representing states, actions, transition probabilities, rewards, and a discount factor. "
        "An agent interacts with an environment to discover an optimal policy that maximizes the expected cumulative discounted return. "
        "Q-learning is an off-policy temporal difference algorithm that iteratively updates state-action values toward the Bellman optimality equation."
    ),
    (
        5,
        "Database Storage Engines and Indexing Structures:\n"
        "Database indexing structures dramatically accelerate data retrieval times at the cost of write amplification. "
        "B-Trees and B+ Trees maintain balanced tree structures on disk with high fan-out, providing O(log N) search, insertion, and range scans. "
        "Log-Structured Merge-Trees (LSM-Trees) optimize write-heavy workloads by appending sequential writes to an in-memory MemTable "
        "before flushing immutable SSTables to disk, which are subsequently compacted in background threads."
    )
]

# Benchmark evaluation dataset with known ground-truth source pages
BENCHMARK_QUERIES = [
    {
        "query": "How do weights get updated using the loss gradient and learning rate?",
        "expected_page": 1,
        "topic": "Gradient Descent Optimization"
    },
    {
        "query": "What is the purpose of learnable filters and max pooling in image processing?",
        "expected_page": 2,
        "topic": "Convolutional Neural Networks"
    },
    {
        "query": "How are Query, Key, and Value matrices combined in scaled dot-product attention?",
        "expected_page": 3,
        "topic": "Transformer Self-Attention"
    },
    {
        "query": "What is an MDP and how does Q-learning update state-action values?",
        "expected_page": 4,
        "topic": "Reinforcement Learning"
    },
    {
        "query": "Why do LSM-Trees use an in-memory MemTable and SSTables for fast writes?",
        "expected_page": 5,
        "topic": "Database Indexing & LSM-Trees"
    },
    {
        "query": "Which architecture uses softmax on Q and K transpose divided by the square root of dimension?",
        "expected_page": 3,
        "topic": "Attention Formula"
    },
    {
        "query": "How does minimizing Mean Squared Error help find optimal linear parameters?",
        "expected_page": 1,
        "topic": "Linear Regression"
    },
    {
        "query": "How do kernels produce feature maps across spatial dimensions?",
        "expected_page": 2,
        "topic": "CNN Kernels"
    }
]


def run_retrieval_evaluation():
    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION: Recall@k Benchmark")
    print("=" * 80)

    # 1. Chunk and index benchmark corpus
    print("\n[1] Chunking and indexing benchmark study corpus...")
    chunks = chunk_document(BENCHMARK_PAGES, chunk_size=400, chunk_overlap=80)
    print(f"    Total pages: {len(BENCHMARK_PAGES)} | Total chunks generated: {len(chunks)}")

    embedder = EmbeddingGenerator.get_instance()
    chunk_embeddings = embedder.embed_chunks(chunks)

    vector_store = FAISSVectorStore()
    vector_store.add_chunks(chunks, chunk_embeddings)
    print(f"    Indexed {vector_store.total_chunks} chunks in FAISS (IndexFlatIP).\n")

    # 2. Evaluate queries
    print("[2] Evaluating test queries...")
    print("-" * 80)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total_queries = len(BENCHMARK_QUERIES)

    results_table = []

    for i, item in enumerate(BENCHMARK_QUERIES, start=1):
        query = item["query"]
        expected_page = item["expected_page"]

        q_vec = embedder.embed_query(query)
        candidates = vector_store.search(q_vec, top_k=10)
        deduped = deduplicate_chunks(candidates, max_overlap_ratio=0.60)[:5]

        retrieved_pages = [chunk.page_number for chunk, _ in deduped]
        top_scores = [round(score, 4) for _, score in deduped]

        hit_1 = expected_page == retrieved_pages[0] if len(retrieved_pages) > 0 else False
        hit_3 = expected_page in retrieved_pages[:3]
        hit_5 = expected_page in retrieved_pages[:5]

        if hit_1:
            hits_at_1 += 1
        if hit_3:
            hits_at_3 += 1
        if hit_5:
            hits_at_5 += 1

        top_page = retrieved_pages[0] if retrieved_pages else None
        top_score = top_scores[0] if top_scores else 0.0

        print(f"Q{i}: \"{query[:65]}...\"")
        print(f"    Expected: Page {expected_page} | Top-1: Page {top_page} (Score: {top_score:.4f})")
        print(f"    Retrieved Pages (Top 5): {retrieved_pages}")
        print(f"    Hits: R@1={hit_1} | R@3={hit_3} | R@5={hit_5}\n")

    # 3. Compute Metrics
    recall_at_1 = (hits_at_1 / total_queries) * 100
    recall_at_3 = (hits_at_3 / total_queries) * 100
    recall_at_5 = (hits_at_5 / total_queries) * 100

    print("=" * 80)
    print("FINAL EVALUATION METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Benchmark Queries : {total_queries}")
    print(f"Recall@1 (Hits@1)       : {hits_at_1}/{total_queries} ({recall_at_1:.1f}%)")
    print(f"Recall@3 (Hits@3)       : {hits_at_3}/{total_queries} ({recall_at_3:.1f}%)")
    print(f"Recall@5 (Hits@5)       : {hits_at_5}/{total_queries} ({recall_at_5:.1f}%)")
    print("=" * 80)
    print("Recall@k measures the percentage of test queries where the ground-truth")
    print("document page appears anywhere in the top-k retrieved results.")
    print("=" * 80)


if __name__ == "__main__":
    run_retrieval_evaluation()
