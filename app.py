import os
import io
import PyPDF2
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PyPDF2 import PdfReader
from PIL import Image
import google.generativeai as genai
from vector_store import add_to_index, search_similar

genai.configure(api_key=os.getenv("Google_API_Key"))
LLM_MODEL = "models/gemini-2.5-flash"

app = FastAPI(title="AI Tutor Agent")

# ----------------------------------
# ROOT PAGE - HTML FORM
# ----------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
    <head>
        <title>AI Tutor Agent</title>
    </head>
    <body>
        <h2>📘 AI Tutor — PDF & Image Multi-Modal Agent</h2>
        <form action="/process" enctype="multipart/form-data" method="post">
            <label>Upload PDF:</label><br>
            <input type="file" name="pdf_file"><br><br>
            <label>Upload Image:</label><br>
            <input type="file" name="img_file"><br><br>
            <label>Enter Topic / Question:</label><br>
            <input type="text" name="query" size="50"><br><br>
            <label>Action:</label>
            <select name="action">
                <option value="summary">Summary</option>
                <option value="flashcards">Flashcards</option>
                <option value="quiz">Quiz</option>
            </select><br><br>
            <input type="submit" value="Run AI Tutor">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ----------------------------------
# PROCESS UPLOAD / QUERY
# ----------------------------------
@app.post("/process", response_class=HTMLResponse)
async def process(
    pdf_file: UploadFile = None,
    img_file: UploadFile = None,
    query: str = Form(...),
    action: str = Form("summary")
):
    output_text = ""

    # PDF ingestion
    if pdf_file:
        content = await pdf_file.read()
        reader = PdfReader(io.BytesIO(content))
        pages = [p.extract_text() for p in reader.pages if p.extract_text()]
        for chunk in pages:
            add_to_index(chunk)
        output_text += f"✅ PDF uploaded and indexed ({len(pages)} chunks)<br>"

    # Image ingestion
    # ... inside async def process( ...

    # FIND THIS SECTION (approx line 70) and REPLACE it with this:
    if img_file:
        file_bytes = await img_file.read()
        
        # Check if it's a PDF
        if img_file.filename.lower().endswith('.pdf'):
            import PyPDF2
            # Read PDF from bytes
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() or ""
            
            # Save to vector store
            add_to_index(text_content)
            output_text += f"✅ PDF '{img_file.filename}' processed and indexed.<br>"
            
        # If it's not a PDF, assume it's an image
        else:
            try:
                img = Image.open(io.BytesIO(file_bytes))
                prompt = "Explain this image for a student."
                response = genai.GenerativeModel("gemini-2.5-flash").generate_content([prompt, img])
                
                explanation = response.text
                add_to_index(explanation)
                output_text += f"✅ Image '{img_file.filename}' processed and indexed.<br>"
            except Exception as e:
                output_text += f"❌ Error processing image: {str(e)}<br>"

    # ... The code continues to 'if action == "quiz":' ...

    # Tutor Agent: retrieval + LLM
    if query.strip():
        context_docs = search_similar(query, k=5)
        context = "\n\n".join([doc for doc, _ in context_docs])

        if action == "summary":
            prompt = f"""
Use the context below to summarize the topic clearly:

{context}

Question: {query}
"""
        elif action == "flashcards":
            prompt = f"""
Use the context below to create 5 flashcards (Q/A):

{context}

Topic: {query}
"""
        elif action == "quiz":
            prompt = f"""
Use the context below to create a 5-question multiple choice quiz:

{context}

Topic: {query}
"""
        else:
            prompt = f"Use the context below to answer:\n{context}\nQuery: {query}"

        response = genai.GenerativeModel(LLM_MODEL).generate_content(prompt)
        output_text += f"<h3>📝 Output:</h3><pre>{response.text}</pre>"

    return HTMLResponse(content=f"<html><body>{output_text}<br><a href='/'>Back</a></body></html>")
