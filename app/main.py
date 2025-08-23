from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.predict import classify_uploaded_pdf
from app.search_engine import SemanticSearch

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all origins (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the semantic search engine (singleton)
semantic_search_engine = SemanticSearch()

@app.post("/classify/")
async def classify(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded PDF file.
    - Expects: A PDF file as 'file'
    - Returns: Classification result
    """
    content = await file.read()
    result = classify_uploaded_pdf(content, filename=file.filename)
    return result

@app.post("/search/")
async def search(query: str = Form(...)):
    """
    Endpoint to perform semantic search.
    - Expects: A 'query' string (form field)
    - Returns: Search results from the semantic search engine
    """
    results = semantic_search_engine.search(query)
    return results
