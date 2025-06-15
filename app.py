import gradio as gr
from pdf_reader import extract_text_from_pdf
from gemini_api import summarize_text, generate_quiz, parse_quiz, answer_question

def process_pdf_summary(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        return "No valid text found in the PDF."
    return summarize_text(text)

def process_pdf_quiz(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    if not text.strip():
        return "No valid text found in the PDF."
    return generate_quiz(text)

def process_question_answer(pdf_file, question):
    context = extract_text_from_pdf(pdf_file)
    if not context.strip():
        return "No valid text found in the PDF."
    return answer_question(context, question)

def generate_and_parse_quiz(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    quiz_text = generate_quiz(text)
    parsed = parse_quiz(quiz_text)
    return quiz_text, parsed

def check_all_answers(q1, q2, q3, parsed):
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
            correct_option = q["options"][correct_idx]
            feedback.append(f"❌ Incorrect. Correct answer: {correct_letter}) \n\n**Explanation:** {explanation}")

    while len(feedback) < 3:
        feedback.append("")
        user_answers[len(feedback)-1] = None

    return (
        [gr.update(value=fb, visible=(i < len(parsed))) for i, fb in enumerate(feedback)] +
        [True] +
        [gr.update(interactive=False, visible=(i < len(parsed))) for i in range(3)] +
        [gr.update(value=f"**Your score: {score}/{len(parsed)}**", visible=True)]
    )

adaptive_css = """
:root {
  --primary: #7c3aed;
  --primary-light: #a78bfa;
  --secondary: #0d9488;
  --tertiary: #3b82f6;
  --accent: #f59e0b;
  --error: #ef4444;
  
  /* Text */
  --text-primary: #1f2937;
  --text-secondary: #4b5563;
  
  /* Backgrounds */
  --bg-surface: #f9fafb;
  --bg-alt: #f3f4f6;
  --bg-input: white;
  
  /* Borders */
  --border-light: rgba(0, 0, 0, 0.1);
  --border-medium: rgba(0, 0, 0, 0.15);
}

.dark {
  --primary: #8b5cf6;
  --primary-light: #c4b5fd;
  --secondary: #2dd4bf;
  --tertiary: #60a5fa;
  --accent: #fbbf24;
  
  /* Text */
  --text-primary: #f9fafb;
  --text-secondary: #d1d5db;
  
  /* Backgrounds */
  --bg-surface: #111827;
  --bg-alt: #1f2937;
  --bg-input: #1a2536;
  
  /* Borders */
  --border-light: rgba(255, 255, 255, 0.1);
  --border-medium: rgba(255, 255, 255, 0.15);
}

.header > div:first-child{
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
.tab-item:not(.selected):hover {
  color: var(--tertiary) !important;
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
.btn-summary:hover {
  transform: translateY(-2px);
  box-shadow: 0 7px 14px color-mix(in srgb, var(--primary), transparent 60%) !important;
}
.btn-quiz:hover {
  transform: translateY(-2px);
  box-shadow: 0 7px 14px color-mix(in srgb, var(--secondary), transparent 60%) !important;
}
.btn-qa:hover {
  transform: translateY(-2px);
  box-shadow: 0 7px 14px color-mix(in srgb, var(--tertiary), transparent 60%) !important;
}

.file-upload {
  border: 2px dashed var(--border-light) !important;
  background: var(--bg-alt) !important;
  position: relative;
  transition: all 0.3s ease !important;
  cursor: pointer;
}
.file-upload:hover {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary), transparent 80%);
}
.file-upload::before {
  content: "";
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: 
    repeating-linear-gradient(
      45deg,
      transparent,
      transparent 7px,
      color-mix(in srgb, var(--primary), transparent 70%) 7px,
      transparent 14px
    );
  opacity: 0;
  transition: opacity 0.3s ease;
  pointer-events: none;
}
.file-upload:hover::before {
  opacity: 0.2;
  animation: moveDots 3s linear infinite;
}
@keyframes moveDots {
  to { background-position: 14px 14px; }
}

input[type="text"], textarea, .dropdown {
  background: var(--bg-input) !important;
  border: 2px solid var(--border-medium) !important;
  padding: 12px 16px !important;
  border-radius: 8px !important;
}
input[type="text"]:focus, textarea:focus {
  border-color: var(--tertiary) !important;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--tertiary), transparent 80%) !important;
}

.output-box, .output-box *, .output-box.prose {
  border: none !important;
  box-shadow: none !important;
}

.output-box {
  min-height: 220px;
  background: var(--bg-surface) !important;
  border-radius: 12px !important;
  border: 2px solid var(--primary) !important;
  position: relative;
  overflow: hidden;
  padding: 5px !important;
}

input[type="radio"] {
  border: 2px solid var(--border-light) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  margin: 8px 0 !important;
  transition: all 0.2s ease !important;
}
input[type="radio"]:hover {
  border-color: var(--tertiary) !important;
}
input[type="radio"]:checked + span {
  background: color-mix(in srgb, var(--tertiary), transparent 90%) !important;
  border-color: var(--tertiary) !important;
}

[data-testid="markdown"]:has(> .feedback) {
  padding: 1rem !important;
  border-radius: 8px !important;
  animation: fadeIn 0.5s ease;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}
.feedback.correct {
  background: color-mix(in srgb, var(--secondary), transparent 85%) !important;
  border-left: 4px solid var(--secondary) !important;
}
.feedback.incorrect {
  background: color-mix(in srgb, var(--error), transparent 85%) !important;
  border-left: 4px solid var(--error) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=adaptive_css) as demo:
    with gr.Column(elem_classes=["header"]):
        gr.Markdown("# 📚 AI Study Assistant")
        gr.Markdown("Gemini-powered learning tools")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Your Document")
            pdf_file = gr.File(label="Upload PDF",elem_classes=["file-upload"], file_types=[".pdf"])

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("📝 Summary", elem_classes=["tab-item"]):
                    summarize_btn = gr.Button("✨ Summarize", elem_classes=["btn-summary"])
                    gr.Markdown("#### Document Summary")
                    summary_output = gr.Markdown(elem_classes=["output-box"])
                    summarize_btn.click(fn=process_pdf_summary, inputs=pdf_file, outputs=summary_output)

                with gr.Tab("🧠 Quiz", elem_classes=["tab-item"]):
                    num_questions_dropdown = gr.Dropdown(
                        choices=["1", "2", "3"],
                        value="3",
                        label="Select number of questions",
                        interactive=True,
                        elem_classes=["dropdown"]
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
                        radio = gr.Radio(choices=[], label="Choose", type="index", interactive=True, visible=False)
                        feedback = gr.Markdown(label="✅ Result", visible=False, elem_classes=["feedback"])
                        quiz_question_boxes.append(question_md)
                        quiz_radio_boxes.append(radio)
                        quiz_feedback_boxes.append(feedback)

                    quiz_submit = gr.Button("✅ Submit All Answers", visible=False)

                    is_quiz_generated = gr.State(False)

                    def quiz_handler(pdf_file, num_q):
                      raw, parsed = generate_and_parse_quiz(pdf_file)
                      n = int(num_q)
                      parsed = parsed[:n]
                      
                      question_updates, radio_updates, feedback_updates = [], [], []

                      for i in range(3):
                          if i < len(parsed):
                              q = parsed[i]
                              question_updates.append(gr.update(value=f"*Q{i+1}. {q['question']}*", visible=True))
                              radio_updates.append(gr.update(choices=q["options"], value=None, visible=True, interactive=True))
                              feedback_updates.append(gr.update(value="", visible=False))
                          else:
                              question_updates.append(gr.update(visible=False))
                              radio_updates.append(gr.update(visible=False))
                              feedback_updates.append(gr.update(visible=False))

                      return [
                          raw,
                          parsed[:n],
                          False,
                          True,
                      ] + question_updates + radio_updates + feedback_updates + [
                          gr.update(visible=True),
                          gr.update(visible=False)
                      ]


                    def toggle_button(is_generated):
                        return gr.update(value="🔁 Retake Quiz" if is_generated else "🧠 Generate Quiz")

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
                with gr.Tab("❓ Q&A", elem_classes=["tab-item"]):
                    question_input = gr.Textbox(label="Your Question", placeholder="Type your question here...")
                    answer_btn = gr.Button("🧐 Get Answer", elem_classes=["btn-qa"])
                    gr.Markdown("#### Answer")
                    answer_output = gr.Markdown(elem_classes=["output-box"])
                    answer_btn.click(fn=process_question_answer, inputs=[pdf_file, question_input], outputs=answer_output)

demo.launch()