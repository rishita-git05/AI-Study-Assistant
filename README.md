# 📚 AI Study Assistant

An AI-powered study tool built with Python and Gradio that can:
- 📄 Summarize PDF documents
- ❓ Generate smart multiple-choice quizzes
- 🧠 Answer user questions based on the content

---

## 🚀 Features
- Extracts clean text from PDFs (skipping headers/footers)
- Summarizes the content using Gemini API
- Generates a structured quiz with explanations
- Accepts follow-up Q&A from users
- Interactive Gradio interface

---

## 🧪 Test PDFs

Here are a couple of sample PDFs you can use:
- [AI Research Notes (short)](https://arxiv.org/pdf/2302.02072.pdf)
- [Machine Learning for Absolute Beginners](https://ia903202.us.archive.org/31/items/python_ebooks_2020/%5BOliver_Theobald%5D_Machine_Learning_for_Absolute_Be.pdf?utm_source=chatgpt.com)

---

## 🔐 API Key Required

To use this app, you must provide your own Gemini API key.

1. Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create a `.env` file in the root folder and add this line:

GEMINI_API_KEY=your_actual_key_here


> ⚠️ Do **NOT** share your `.env` or actual key publicly.

---

## 🛠 Tech Stack

- Python
- Gradio
- Gemini API
- pdfplumber
- dotenv
- regex

---

## 📌 Notes

- Quiz shows score and explanations after submission
- You can retake the quiz once generated
- PDF summarization skips non-informative sections
- Dark/light mode compatible

---

## 📜 License


This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

- ✅ Free to **view and share** for educational purposes.
- ❌ **Commercial use**, **modification**, or **redistribution** of the code is **not allowed**.
- ⚠️ You must also adhere to the [Google Gemini API Terms of Service](https://ai.google.dev/terms).
