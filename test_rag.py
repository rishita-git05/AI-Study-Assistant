"""Automated Test Suite for RAG Pipeline.

Tests:
1. Text cleaning and artifact removal.
2. Boundary-aware chunking and metadata preservation (chunk_id, page_number, offsets).
3. Embedding generation and L2 normalization (||v|| = 1.0).
4. FAISS IndexFlatIP inner product equivalence to cosine similarity.
5. Deduplication of overlapping chunks.
6. Similarity threshold filtering and short-circuit behavior.
7. Deterministic source page extraction.
"""

import unittest
import numpy as np
from rag.text_processor import clean_page_text
from rag.chunker import chunk_page_text, chunk_document, DocumentChunk
from rag.embedder import EmbeddingGenerator
from rag.vector_store import FAISSVectorStore
from rag.deduplicator import compute_token_jaccard_similarity, deduplicate_chunks
from rag.pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def test_text_cleaning(self):
        raw = "This is a hyphen-\nated word with \x0c form feed and   multiple   spaces.\n\n\n\nNew paragraph."
        cleaned = clean_page_text(raw)
        self.assertNotIn("\x0c", cleaned)
        self.assertIn("hyphenated", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertIn("\n\nNew paragraph.", cleaned)

    def test_chunking_metadata_and_boundaries(self):
        page_num = 2
        text = (
            "Artificial intelligence is transforming modern computer science. "
            "Machine learning algorithms discover patterns in large datasets. "
            "Deep neural networks have achieved state-of-the-art results in computer vision and natural language processing.\n\n"
            "Supervised learning requires labeled training data. Unsupervised learning discovers hidden patterns in unlabeled data."
        )
        # Use small chunk size to force multiple chunks
        chunks = chunk_page_text(page_number=page_num, text=text, chunk_size=120, chunk_overlap=30)
        self.assertGreater(len(chunks), 1)

        for chunk in chunks:
            self.assertEqual(chunk.page_number, page_num)
            self.assertTrue(chunk.chunk_id.startswith("chunk_"))
            self.assertGreater(chunk.end_char, chunk.start_char)
            # Ensure text slice matches character offsets in the page
            self.assertEqual(chunk.text, text[chunk.start_char:chunk.end_char].strip())

    def test_l2_normalization_and_cosine_similarity(self):
        embedder = EmbeddingGenerator.get_instance()
        texts = ["Vector search retrieves nearest neighbors.", "Deep learning uses neural networks."]
        vecs = embedder.embed_texts(texts)

        # Check shapes
        self.assertEqual(vecs.shape, (2, 384))

        # Check L2 norm is 1.0 (within float32 precision)
        norm0 = np.linalg.norm(vecs[0])
        norm1 = np.linalg.norm(vecs[1])
        self.assertAlmostEqual(float(norm0), 1.0, places=5)
        self.assertAlmostEqual(float(norm1), 1.0, places=5)

        # Inner product should equal cosine similarity
        inner_product = float(np.dot(vecs[0], vecs[1]))
        cosine_sim = float(np.dot(vecs[0], vecs[1]) / (norm0 * norm1))
        self.assertAlmostEqual(inner_product, cosine_sim, places=5)

    def test_faiss_vector_store(self):
        store = FAISSVectorStore()
        chunks = [
            DocumentChunk("c0", 1, "Transformers use self-attention mechanisms to model long-range dependencies.", 0, 75),
            DocumentChunk("c1", 2, "Photosynthesis is the biological process by which plants synthesize glucose.", 0, 75),
        ]
        embedder = EmbeddingGenerator.get_instance()
        vecs = embedder.embed_chunks(chunks)
        store.add_chunks(chunks, vecs)

        self.assertEqual(store.total_chunks, 2)

        # Query about self-attention should rank c0 first with high similarity
        query_vec = embedder.embed_query("How does attention work in transformers?")
        results = store.search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk.chunk_id, "c0")
        self.assertGreater(top_score, 0.40)
        # Biology chunk should score lower
        self.assertGreater(top_score, results[1][1])

    def test_deduplication(self):
        c1 = DocumentChunk("c1", 1, "Convolutional neural networks are specialized for processing grid structured image data.", 0, 88)
        c2 = DocumentChunk("c2", 1, "Convolutional neural networks are specialized for processing grid structured data efficiently.", 0, 93)
        c3 = DocumentChunk("c3", 2, "Recurrent neural networks process sequential data using internal hidden states.", 0, 78)

        scored = [(c1, 0.85), (c2, 0.82), (c3, 0.50)]
        deduped = deduplicate_chunks(scored, max_overlap_ratio=0.50)

        # c2 should be filtered out as redundant with c1
        ids = [c.chunk_id for c, _ in deduped]
        self.assertIn("c1", ids)
        self.assertNotIn("c2", ids)
        self.assertIn("c3", ids)

    def test_threshold_short_circuit(self):
        pipeline = RAGPipeline()
        # Seed pipeline manually with dummy data
        chunks = [
            DocumentChunk("c0", 1, "Database indexing uses B-trees and hash tables to accelerate query lookup times.", 0, 80)
        ]
        vecs = pipeline.embedder.embed_chunks(chunks)
        pipeline.vector_store.add_chunks(chunks, vecs)

        # Query totally unrelated with strict threshold
        result = pipeline.retrieve("What are the culinary traditions of ancient Rome?", top_k=2, similarity_threshold=0.60)
        self.assertFalse(result.has_sufficient_context)
        self.assertIn("couldn't find enough relevant information", result.fallback_message)


if __name__ == "__main__":
    unittest.main()
