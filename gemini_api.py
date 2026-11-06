"""Gemini API integration module for grounded generation, summarization, and quiz creation.

Enforces strict anti-hallucination prompting for RAG query-answering,
supports hierarchical map-reduce summarization for large documents,
and preserves structured quiz generation.
"""

import os
import re
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from rag.config import MAX_SUMMARY_CHUNK_CHARS

# Load environment variables
load_dotenv()

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")

def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    """Returns configured GenerativeModel instance with error checking."""
    current_key = os.getenv("GEMINI_API_KEY")
    if not current_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set your Gemini API key in the .env file."
        )
    genai.configure(api_key=current_key)
    return genai.GenerativeModel(model_name)


# Grounded RAG Prompt Template
GROUNDED_QA_PROMPT = """You are an AI Study Assistant providing accurate, factual answers based strictly on retrieved document excerpts.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the clear facts mentioned directly in the "Retrieved Context" section below.
2. Do NOT extrapolate, speculate, or invent any information that is not directly stated in the context.
3. Do NOT use unsupported information from your general pre-trained knowledge.
4. If the retrieved context does not contain enough information to answer the question completely, explicitly state:
   "Based on the uploaded document, I could not find enough information to answer this question."
5. Do NOT invent or fabricate citations or page numbers.
6. Present your answer clearly, concisely, and objectively.

Retrieved Context:
{context}

Question:
{question}

Grounded Answer:"""


def generate_grounded_answer(context: str, question: str) -> str:
    """Generates a strictly grounded answer using Gemini based solely on retrieved context.
    
    Args:
        context: Formatted string of retrieved chunks with page/chunk identifiers.
        question: User's question.

    Returns:
        Generated grounded answer string.
    """
    if not context or not context.strip():
        return "I couldn't find enough relevant information in the uploaded document to answer this question."

    try:
        model = get_gemini_model()
        prompt = GROUNDED_QA_PROMPT.format(context=context.strip(), question=question.strip())
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2  # Low temperature for strict factual grounding
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


def answer_question(context: str, question: str) -> str:
    """Backward-compatible adapter for question answering."""
    return generate_grounded_answer(context, question)


# ============================================================================
# Document Summarization (Single-Pass & Hierarchical Map-Reduce)
# ============================================================================

def summarize_text(text: str) -> str:
    """Summarizes document text.
    
    For standard documents, performs single-pass summarization.
    For large documents exceeding MAX_SUMMARY_CHUNK_CHARS, applies a hierarchical
    map-reduce approach (summarize sections, then synthesize executive summary)
    to prevent context window bloat and attention degradation.
    """
    cleaned = text.strip()
    if not cleaned:
        return "No text available to summarize."

    try:
        model = get_gemini_model()

        # Check if text exceeds single-pass threshold
        if len(cleaned) > MAX_SUMMARY_CHUNK_CHARS:
            print(f"[gemini_api] Document text ({len(cleaned)} chars) exceeds threshold. Applying hierarchical summarization.")
            return _hierarchical_summarize(cleaned, model)

        # Single-pass summarization
        prompt = f"""You are an expert academic tutor. Provide a comprehensive, structured summary of the following document.

Document Content:
{cleaned}

Format your summary with:
- **Executive Overview**: 2-3 sentences capturing the core theme.
- **Key Concepts & Findings**: Bullet points detailing major topics and takeaways.
- **Conclusions / Implications**: Final concluding insights.
"""
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error during summarization: {e}"


def _hierarchical_summarize(text: str, model) -> str:
    """Map-reduce summarization: summarizes section chunks, then combines them."""
    chunk_size = 15000
    sections = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    section_summaries = []

    print(f"[gemini_api] Summarizing {len(sections)} sections in parallel/sequence...")
    for idx, sec in enumerate(sections, start=1):
        sec_prompt = f"Summarize the key points of Section {idx} of this study document concisely:\n\n{sec}"
        try:
            res = model.generate_content(sec_prompt)
            section_summaries.append(f"### Section {idx} Summary:\n{res.text.strip()}")
        except Exception as e:
            section_summaries.append(f"### Section {idx} Summary:\n[Failed to summarize section: {e}]")

    combined_sections = "\n\n".join(section_summaries)
    reduce_prompt = f"""You are an expert academic tutor. Below are section-by-section summaries of a large document.
Synthesize them into a single, cohesive, high-level executive summary:

{combined_sections}

Format your summary with:
- **Executive Overview**
- **Core Topics & Key Takeaways**
- **Final Summary**
"""
    final_res = model.generate_content(reduce_prompt)
    return final_res.text.strip()


# ============================================================================
# Quiz Generation & Parsing
# ============================================================================

def generate_quiz(text: str) -> str:
    """Generates a structured multiple-choice quiz from document text."""
    cleaned = text.strip()
    if not cleaned:
        return "No text available to generate a quiz."

    # If text is very large, sample up to 15,000 characters to keep prompt focused
    if len(cleaned) > 15000:
        cleaned = cleaned[:15000]

    prompt = f"""Create a short multiple-choice quiz (3 questions) based on this study content:

{cleaned}

Each question must strictly have:
- A question
- 3 options labeled a), b), and c)
- The correct answer (like: Answer: a)
- A brief explanation (like: Explanation: ...)

Format:
Q1. Question text?
a) Option A
b) Option B
c) Option C
Answer: a
Explanation: Because...

Only follow this exact format.
"""
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating quiz: {e}"


def parse_quiz(quiz_text: str) -> List[dict]:
    """Parses raw quiz output into structured JSON questions."""
    pattern = re.compile(
        r"Q\d+\.\s*(.*?)\n"                # question
        r"a\)\s*(.*?)\n"                   # option a
        r"b\)\s*(.*?)\n"                   # option b
        r"c\)\s*(.*?)\n"                   # option c
        r"Answer:\s*([abc])\n"             # answer
        r"Explanation:\s*(.*?)(?:\n|$)",   # explanation
        re.DOTALL
    )

    matches = pattern.findall(quiz_text)
    questions = []
    for q, a, b, c, ans, explanation in matches:
        questions.append({
            "question": q.strip(),
            "options": [a.strip(), b.strip(), c.strip()],
            "answer": ans.strip(),
            "explanation": explanation.strip()
        })
    return questions
