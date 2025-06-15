import pdfplumber

def extract_text_from_pdf(file_obj):
    if file_obj is None:
        return ""
    try:
        with pdfplumber.open(file_obj.name) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
