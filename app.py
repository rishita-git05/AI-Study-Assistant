"""AI Study Assistant - RAG-Powered Gradio Application.

Features:
- Offline Ingestion & FAISS Vector Indexing with real-time status.
- Genuine RAG Q&A with dense semantic retrieval, similarity thresholding, and deduplication.
- Deterministic source page attribution.
- Retrieval Inspection & Debug inspector displaying rank, chunk ID, page number, cosine similarity, and snippets.
- Global document summarization with hierarchical map-reduce fallback.
- Interactive multi-choice quiz generation with automated scoring.
"""

from typing import Tuple, List
import gradio as gr
from rag.pipeline import RAGPipeline, RetrievalResult
from rag.config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD
from gemini_api import (
    generate_grounded_answer,
    summarize_text,
    generate_quiz,
    parse_quiz
)

# Initialize application-level RAG Pipeline
rag_pipeline = RAGPipeline()


def handle_pdf_upload(pdf_file) -> str:
    """Triggered whenever a user uploads or replaces a PDF document.
    
    Executes Stage 1 (Offline Ingestion & Indexing Pipeline).
    """
    if pdf_file is None:
        rag_pipeline.clear()
        return "📁 Please upload a PDF document to begin."

    result = rag_pipeline.index_document(pdf_file)
    if not result.get("success"):
        return f"⚠️ **Indexing Failed:** {result.get('error')}"

    return (
        f"✅ **Document Indexed Successfully!**\n\n"
        f"• **Pages Extracted:** {result['total_pages']}\n"
        f"• **Text Chunks Created:** {result['total_chunks']}\n"
        f"• **FAISS Vector Index:** {result['indexed_vectors']} dense vectors (384-dim, L2-normalized)\n\n"
        f"_Ready for grounded RAG Q&A, summarization, and quiz generation._"
    )


def process_rag_question_answer(
    pdf_file,
    question: str,
    top_k: int,
    similarity_threshold: float
) -> Tuple[str, str, str]:
    """Executes Stage 2 (Online Query-Time RAG Pipeline).
    
    Query -> Dense Embedding -> FAISS Search -> Deduplication ->
    Threshold Filter -> Grounded Gemini Generation -> Source Attribution.
    """
    if pdf_file is None:
        return (
            "⚠️ Please upload a PDF document before asking a question.",
            "📌 **Sources:** None",
            "No document indexed."
        )

    # If the document hasn't been indexed yet, index it first
    if not rag_pipeline.is_indexed:
        idx_res = rag_pipeline.index_document(pdf_file)
        if not idx_res.get("success"):
            return (
                f"⚠️ Could not index document: {idx_res.get('error')}",
                "📌 **Sources:** None",
                "Indexing failed."
            )

    # 1. Semantic Retrieval with Deduplication & Thresholding
    retrieval_result: RetrievalResult = rag_pipeline.retrieve(
        query=question,
        top_k=int(top_k),
        similarity_threshold=float(similarity_threshold)
    )

    # 2. Build Debug Inspector Table
    debug_markdown = _format_debug_table(retrieval_result, similarity_threshold)

    # 3. Check if threshold filtering permitted generation
    if not retrieval_result.has_sufficient_context:
        answer_text = (
            f"ℹ️ **No Sufficient Context Found**\n\n"
            f"{retrieval_result.fallback_message}\n\n"
            f"> *Note: No retrieved chunks met the similarity threshold of {similarity_threshold:.2f}. "
            f"LLM generation was short-circuited to prevent hallucination.*"
        )
        sources_text = "📌 **Sources:** None (No chunks passed relevance threshold)"
        return answer_text, sources_text, debug_markdown

    # 4. Grounded Gemini Generation using ONLY retrieved context
    grounded_answer = generate_grounded_answer(
        context=retrieval_result.formatted_context,
        question=question
    )

    # 5. Deterministic Source Attribution from Chunk Metadata
    if retrieval_result.sources:
        pages_str = ", ".join([f"Page {p}" for p in retrieval_result.sources])
        sources_text = f"📌 **Sources:** {pages_str}"
    else:
        sources_text = "📌 **Sources:** Not specified"

    return grounded_answer, sources_text, debug_markdown


def _format_debug_table(result: RetrievalResult, threshold: float) -> str:
    """Formats retrieved chunk diagnostic metadata into an inspection view."""
    if not result.debug_chunks:
        return "_No candidate chunks were retrieved._"

    lines = [
        "### 🔍 Retrieval Inspection & Diagnostics\n",
        f"**Query:** *\"{result.query}\"* | **Threshold Cutoff:** `{threshold:.2f}`\n",
        "| Rank | Chunk ID | Source Page | Cosine Similarity | Threshold Status | Chunk Text Snippet |",
        "| :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for c in result.debug_chunks:
        status = "✅ Passed" if c.passed_threshold else "❌ Filtered Out"
        clean_snippet = c.text.replace("\n", " ").replace("|", "\\|")
        snippet = (clean_snippet[:100] + "...") if len(clean_snippet) > 100 else clean_snippet
        lines.append(
            f"| #{c.rank} | `{c.chunk_id}` | Page {c.page_number} | **{c.score:.4f}** | {status} | {snippet} |"
        )

    lines.append("\n> **Mathematical Rationale:**")
    lines.append(
        "> All vectors are L2-normalized ($\\|\\mathbf{v}\\|_2 = 1.0$), so the FAISS `IndexFlatIP` "
        "inner product is mathematically identical to **Cosine Similarity**: "
        "$\\mathbf{u} \\cdot \\mathbf{v} = \\cos(\\theta)$."
    )

    return "\n".join(lines)


def process_pdf_summary(pdf_file) -> str:
    """Generates document summary with hierarchical map-reduce fallback."""
    if pdf_file is None:
        return "Please upload a valid PDF document."

    if not rag_pipeline.is_indexed:
        rag_pipeline.index_document(pdf_file)

    full_text = rag_pipeline.get_document_full_text()
    if not full_text.strip():
        return "No valid text found in the PDF."

    return summarize_text(full_text)


def generate_and_parse_quiz(pdf_file) -> Tuple[str, List[dict]]:
    """Generates and parses interactive multiple-choice quiz."""
    if pdf_file is None:
        return "Please upload a valid PDF document.", []

    if not rag_pipeline.is_indexed:
        rag_pipeline.index_document(pdf_file)

    full_text = rag_pipeline.get_document_full_text()
    if not full_text.strip():
        return "No valid text found in the PDF.", []

    quiz_text = generate_quiz(full_text)
    parsed = parse_quiz(quiz_text)
    return quiz_text, parsed


def check_all_answers(q1, q2, q3, parsed):
    """Scores quiz selections against parsed ground truth."""
    user_answers = [q1, q2, q3]
    score = 0
    feedback = []

    for i, q in enumerate(parsed):
        correct_letter = q["answer"]
        correct_idx = {"a": 0, "b": 1, "c": 2}[correct_letter]
        selected = user_answers[i]
        explanation = q.get("explanation", "No explanation provided.")

        if selected is None:
            feedback.append("⚠️ Not answered")
        elif selected == correct_idx:
            feedback.append(f"✅ Correct!\n\n**Explanation:** {explanation}")
            score += 1
        else:
            feedback.append(
                f"❌ Incorrect. Correct answer: {correct_letter}) \n\n**Explanation:** {explanation}"
            )

    while len(feedback) < 3:
        feedback.append("")
        user_answers[len(feedback) - 1] = None

    return (
        [gr.update(value=fb, visible=(i < len(parsed))) for i, fb in enumerate(feedback)] +
        [True] +
        [gr.update(interactive=False, visible=(i < len(parsed))) for i in range(3)] +
        [gr.update(value=f"**Your score: {score}/{len(parsed)}**", visible=True)]
    )


# ============================================================================
# Gradio UI Design
# ============================================================================

adaptive_css = """
:root {
  --primary: #7c3aed;
  --primary-light: #a78bfa;
  --secondary: #0d9488;
  --tertiary: #3b82f6;
  --accent: #f59e0b;
  --error: #ef4444;
  --bg-surface: #f9fafb;
  --bg-alt: #f3f4f6;
  --bg-input: white;
  --border-light: rgba(0, 0, 0, 0.1);
  --border-medium: rgba(0, 0, 0, 0.15);
}

.dark {
  --primary: #8b5cf6;
  --primary-light: #c4b5fd;
  --secondary: #2dd4bf;
  --tertiary: #60a5fa;
  --accent: #fbbf24;
  --bg-surface: #111827;
  --bg-alt: #1f2937;
  --bg-input: #1a2536;
  --border-light: rgba(255, 255, 255, 0.1);
  --border-medium: rgba(255, 255, 255, 0.15);
}

.header > div:first-child {
  font-size: 2.5rem;
  background: linear-gradient(45deg, var(--primary), var(--tertiary));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  position: relative;
  padding-bottom: 10px;
  margin-bottom: 0.3rem;
}

.header > div:first-child::after {
  content: "";
  position: absolute;
  left: 0;
  bottom: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, var(--primary), var(--tertiary));
  border-radius: 3px;
}

.tab-nav {
  border-bottom: none !important;
  gap: 5px;
}

.tab-item {
  padding: 12px 20px !important;
  border-radius: 12px 12px 0 0 !important;
  transition: all 0.3s ease !important;
}

.tab-item.selected {
  background: var(--bg-alt) !important;
  color: var(--primary) !important;
  font-weight: 600;
  box-shadow: 0 -3px 0 var(--primary) inset,
              0 2px 10px rgba(0,0,0,0.1);
}

.btn-summary, .btn-quiz, .btn-qa {
  font-weight: 600 !important;
  letter-spacing: 0.5px;
  border: none !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.btn-summary {
  background: linear-gradient(135deg, var(--primary), var(--primary-light)) !important;
}

.btn-quiz {
  background: linear-gradient(135deg, var(--secondary), #6ee7b7) !important;
}

.btn-qa {
  background: linear-gradient(135deg, var(--tertiary), #93c5fd) !important;
}

.btn-summary:hover, .btn-quiz:hover, .btn-qa:hover {
  transform: translateY(-2px);
}

.file-upload {
  border: 2px dashed var(--border-light) !important;
  background: var(--bg-alt) !important;
  border-radius: 12px !important;
}

.status-box {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-medium) !important;
  border-radius: 8px !important;
  padding: 12px !important;
  font-size: 0.9rem;
}

.source-badge {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--primary);
  background: var(--bg-alt);
  padding: 8px 14px;
  border-radius: 8px;
  border-left: 4px solid var(--primary);
  margin-bottom: 10px;
}

.output-box {
  min-height: 180px;
  background: var(--bg-surface) !important;
  border-radius: 12px !important;
  border: 2px solid var(--border-medium) !important;
  padding: 12px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=adaptive_css) as demo:
    with gr.Column(elem_classes=["header"]):
        gr.Markdown("# 📚 AI Study Assistant (RAG Engine)")
        gr.Markdown("Dense Semantic Retrieval with FAISS & Grounded Gemini Generation")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### 📄 Document Ingestion")
            pdf_file = gr.File(
                label="Upload Study PDF",
                elem_classes=["file-upload"],
                file_types=[".pdf"]
            )
            doc_status = gr.Markdown(
                "📁 Upload a PDF to trigger indexing.",
                elem_classes=["status-box"]
            )
            pdf_file.change(fn=handle_pdf_upload, inputs=pdf_file, outputs=doc_status)

        with gr.Column(scale=3):
            with gr.Tabs():
                # TAB 1: RAG Q&A
                with gr.Tab("❓ Grounded Q&A (RAG)", elem_classes=["tab-item"]):
                    gr.Markdown("#### Semantic Retrieval & Grounded Answering")
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask a factual question about the uploaded document...",
                        lines=2
                    )

                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=DEFAULT_TOP_K,
                            step=1,
                            label="Top-k Chunks Retrieved",
                            info="Number of nearest chunks retrieved from FAISS"
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.10,
                            maximum=0.80,
                            value=SIMILARITY_THRESHOLD,
                            step=0.05,
                            label="Similarity Threshold",
                            info="Minimum Cosine Similarity cutoff to prevent hallucination"
                        )

                    answer_btn = gr.Button("🧐 Ask with RAG", elem_classes=["btn-qa"])

                    gr.Markdown("#### Grounded Answer")
                    source_output = gr.Markdown(
                        "📌 **Sources:** No query processed yet.",
                        elem_classes=["source-badge"]
                    )
                    answer_output = gr.Markdown(elem_classes=["output-box"])

                    with gr.Accordion("🔍 Inspect Retrieved Chunks & Similarity Scores (RAG Debugger)", open=False):
                        debug_output = gr.Markdown("_Submit a question to inspect retrieved FAISS chunks._")

                    answer_btn.click(
                        fn=process_rag_question_answer,
                        inputs=[pdf_file, question_input, top_k_slider, threshold_slider],
                        outputs=[answer_output, source_output, debug_output]
                    )

                # TAB 2: Document Summary
                with gr.Tab("📝 Summary", elem_classes=["tab-item"]):
                    gr.Markdown("#### Global Document Synthesis")
                    summarize_btn = gr.Button("✨ Summarize Document", elem_classes=["btn-summary"])
                    summary_output = gr.Markdown(elem_classes=["output-box"])
                    summarize_btn.click(
                        fn=process_pdf_summary,
                        inputs=pdf_file,
                        outputs=summary_output
                    )

                # TAB 3: Quiz Generation
                with gr.Tab("🧠 Interactive Quiz", elem_classes=["tab-item"]):
                    gr.Markdown("#### Knowledge Check")
                    num_questions_dropdown = gr.Dropdown(
                        choices=["1", "2", "3"],
                        value="3",
                        label="Select number of questions",
                        interactive=True
                    )
                    quiz_btn = gr.Button("🧠 Generate Quiz", elem_classes=["btn-quiz"])
                    quiz_text_output = gr.Textbox(label="Generated Quiz (raw)", visible=False)

                    quiz_question_boxes = []
                    quiz_radio_boxes = []
                    quiz_feedback_boxes = []
                    parsed_quiz_data = gr.State([])
                    quiz_submitted = gr.State(False)
                    quiz_score_output = gr.Markdown(visible=False)

                    for _ in range(3):
                        question_md = gr.Markdown(visible=False)
                        radio = gr.Radio(
                            choices=[],
                            label="Choose",
                            type="index",
                            interactive=True,
                            visible=False
                        )
                        feedback = gr.Markdown(visible=False)
                        quiz_question_boxes.append(question_md)
                        quiz_radio_boxes.append(radio)
                        quiz_feedback_boxes.append(feedback)

                    quiz_submit = gr.Button("✅ Submit All Answers", visible=False)
                    is_quiz_generated = gr.State(False)

                    def quiz_handler(pdf, num_q):
                        raw, parsed = generate_and_parse_quiz(pdf)
                        n = int(num_q)
                        parsed = parsed[:n]

                        q_updates, r_updates, fb_updates = [], [], []
                        for i in range(3):
                            if i < len(parsed):
                                q = parsed[i]
                                q_updates.append(gr.update(value=f"*Q{i+1}. {q['question']}*", visible=True))
                                r_updates.append(gr.update(choices=q["options"], value=None, visible=True, interactive=True))
                                fb_updates.append(gr.update(value="", visible=False))
                            else:
                                q_updates.append(gr.update(visible=False))
                                r_updates.append(gr.update(visible=False))
                                fb_updates.append(gr.update(visible=False))

                        return [
                            raw,
                            parsed[:n],
                            False,
                            True
                        ] + q_updates + r_updates + fb_updates + [
                            gr.update(visible=True),
                            gr.update(visible=False)
                        ]

                    def toggle_button(is_gen):
                        return gr.update(value="🔁 Retake Quiz" if is_gen else "🧠 Generate Quiz")

                    quiz_btn.click(
                        fn=quiz_handler,
                        inputs=[pdf_file, num_questions_dropdown],
                        outputs=[quiz_text_output, parsed_quiz_data, quiz_submitted, is_quiz_generated]
                                + quiz_question_boxes + quiz_radio_boxes + quiz_feedback_boxes
                                + [quiz_submit, quiz_score_output]
                    )

                    is_quiz_generated.change(
                        fn=toggle_button,
                        inputs=is_quiz_generated,
                        outputs=quiz_btn
                    )

                    quiz_submit.click(
                        fn=check_all_answers,
                        inputs=quiz_radio_boxes + [parsed_quiz_data],
                        outputs=quiz_feedback_boxes + [quiz_submitted] + quiz_radio_boxes + [quiz_score_output]
                    )

if __name__ == "__main__":
    demo.launch()