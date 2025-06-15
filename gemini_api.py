import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load environment variables from .env
load_dotenv()

# Read API key from .env
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# Use Gemini 1.5 Flash
model = genai.GenerativeModel("gemini-1.5-flash-latest")


def summarize_text(text):
    prompt = f"Summarize the following text:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text


def generate_quiz(text):
    prompt = f"""Create a short multiple-choice quiz (2-3 questions) from this content:

{text}

Each question must have:
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
    response = model.generate_content(prompt)
    return response.text


def parse_quiz(quiz_text):
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


def answer_question(context, question):
    prompt = f"""You are a helpful AI tutor. Based on the following study material, answer the question clearly.

Study material:
{context}

Question:
{question}
"""
    response = model.generate_content(prompt)
    return response.text
