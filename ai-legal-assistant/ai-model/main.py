import os
import tempfile
import hashlib
import re
from typing import Dict
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
# from langchain_community.vectorstores.utils import distance
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.clause_extractor import ClauseExtractor

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to save vectorstores
VECTORSTORE_DIR = "vectorstores"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# -------------------------------
# Caches
# -------------------------------
vectorstore_cache: Dict[str, VectorStore] = {}
legal_docs_store: Dict[str, VectorStore] = {}  # For /ask-existing


# -------------------------------
# Utility: File hash
# -------------------------------
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# -------------------------------
# Utility: Clean AI response
# -------------------------------
def clean_ai_response(response: str) -> str:
    """Clean AI response by removing excessive formatting symbols."""
    # Remove excessive asterisks and bullet points
    cleaned = re.sub(r'\*{2,}', '', response)  # Remove multiple asterisks
    cleaned = re.sub(r'^\s*[\*\-\•]\s*', '', cleaned, flags=re.MULTILINE)  # Remove bullet points
    cleaned = re.sub(r'\*([^\*]+)\*', r'\1', cleaned)  # Remove single asterisk emphasis
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Reduce multiple newlines
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)  # Remove markdown headers
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)  # Remove code formatting
    return cleaned.strip()

# -------------------------------
# Utility: Create FAISS vectorstore safely
# -------------------------------
def create_faiss_vectorstore_safe(chunks, embeddings, name: str = None):
    try:
        vs = FAISS.from_documents(chunks, embeddings)
        if name:
            save_path = os.path.join(VECTORSTORE_DIR, name)
            vs.save_local(save_path)
        return vs
    except Exception as e:
        print(f"⚠️ Failed to embed documents: {e}")
        return None


# smart chunk splitting
def smart_chunk_splitter(docs):
    final_chunks = []

    for doc in docs:
        length = len(doc.page_content)

        # Dynamically decide chunk size and overlap
        if length < 1000:
            chunk_size = 400
            chunk_overlap = 50
        elif length < 3000:
            chunk_size = 700
            chunk_overlap = 100
        else:
            chunk_size = 1000
            chunk_overlap = 120

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Always split each document individually
        chunks = splitter.split_documents([doc])
        final_chunks.extend(chunks)

    return final_chunks



# -------------------------------
# Startup: Preload legal docs
# -------------------------------
@app.on_event("startup")
async def preload_legal_documents():
    print("🔍 Preloading legal documents...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    predefined_pdfs = {
        "Guide to Litigation in India": "data/Guide-to-Litigation-in-India.pdf",
        "Legal Compliance & Corporate Laws": "data/Legal-Compliance-Corporate-Laws.pdf",
        "legaldoc": "data/legaldoc.pdf",
        "Constitution of India": "data/constitution_of_india.pdf",
        "IPC": "data/penal_code.pdf",
        "Format": "data/format.pdf"
    }

    for name, path in predefined_pdfs.items():
        save_path = os.path.join(VECTORSTORE_DIR, name)

        if os.path.exists(save_path):
            print(f"✅ Loading cached vectorstore for: {name}")
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        else:
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = smart_chunk_splitter(docs)

            for chunk in chunks:
                chunk.metadata["source"] = name

            vectorstore = create_faiss_vectorstore_safe(chunks, embeddings, name)

        if vectorstore:
            legal_docs_store[name] = vectorstore

    print("✅ Legal documents preloaded.")

# -------------------------------
# /ask-existing: Ask from preloaded legal docs
# -------------------------------
@app.post("/ask-existing")
async def ask_from_existing(query: str = Form(...)):
    if not legal_docs_store:
        return {"error": "Legal documents not loaded yet."}

    all_matches = []
    for name, vectorstore in legal_docs_store.items():
        results = vectorstore.similarity_search_with_score(query, k=5)
        for doc, score in results:
            if doc and score is not None:
                all_matches.append({
                    "source": doc.metadata.get("source", name),
                    "content": doc.page_content,
                    "score": score
                })

    if not all_matches:
        return {"error": "No relevant information found."}

    from collections import defaultdict
    source_scores = defaultdict(list)
    for match in all_matches:
        source_scores[match["source"]].append(match["score"])

    best_source = min(source_scores, key=lambda s: sum(source_scores[s]) / len(source_scores[s]))
    best_chunks = [m["content"] for m in all_matches if m["source"] == best_source]
    combined_text = "\n\n".join(best_chunks)

    prompt = f"""
You are a legal assistant. Use the following legal document excerpts to answer the user's question.

---DOCUMENT EXCERPTS---
{combined_text}
-----------------------

Question: {query}

Provide a legally accurate, helpful, and context-aware answer.
"""

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    cleaned_answer = clean_ai_response(answer)

    return {"answer": cleaned_answer, "source": best_source}

# -------------------------------
# /ask-upload: Upload PDF & Ask
# -------------------------------
@app.post("/ask-upload")
async def ask_from_uploaded(query: str = Form(...), file: UploadFile = None):
    if file is None:
        return {"error": "No file uploaded."}

    file_bytes = await file.read()
    file_id = file_hash(file_bytes)
    save_path = os.path.join(VECTORSTORE_DIR, file_id)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    if file_id in vectorstore_cache:
        vectorstore = vectorstore_cache[file_id]
    elif os.path.exists(save_path):
        vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        vectorstore_cache[file_id] = vectorstore
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        chunks = smart_chunk_splitter(docs)

        vectorstore = create_faiss_vectorstore_safe(chunks, embeddings, name=file_id)
        if vectorstore:
            vectorstore_cache[file_id] = vectorstore

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    result = qa_chain.run(query)
    cleaned_result = clean_ai_response(result)

    return {"answer": cleaned_result, "file_id": file_id}

# -------------------------------
# /chat: General chat endpoint
# -------------------------------
@app.post("/chat")
async def general_chat(query: str = Form(...)):
    """General chat endpoint for conversational AI without specific document context"""
    
    prompt = f"""
You are a helpful AI legal assistant. Provide professional, accurate, and helpful legal guidance.
Be conversational but maintain professionalism. If a question requires specific legal documents 
or analysis, suggest the user upload a document or use the legal database.

User Question: {query}

Provide a helpful, informative response:
"""

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, 'content') else str(response)
    cleaned_answer = clean_ai_response(answer)

    return {"response": cleaned_answer}

# -------------------------------
# /save-chat: Save chat for history
# -------------------------------
@app.post("/save-chat")
async def save_chat(chat_id: str = Form(...), user_message: str = Form(...), ai_response: str = Form(...)):
    """Save a chat conversation for history"""
    # This endpoint will be called by the frontend to save chat messages
    # For now, we'll just return success - the actual saving is handled by the Next.js backend
    return {"success": True, "chat_id": chat_id}

# -------------------------------
# /ask-context: Ask using file_id
# -------------------------------
@app.post("/ask-context")
async def ask_from_context(query: str = Form(...), file_id: str = Form(...)):
    if file_id not in vectorstore_cache:
        save_path = os.path.join(VECTORSTORE_DIR, file_id)
        if os.path.exists(save_path):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=GEMINI_API_KEY
            )
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore_cache[file_id] = vectorstore
        else:
            return {"error": "Context not found. Please upload the file first."}

    vectorstore = vectorstore_cache[file_id]

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    result = qa_chain.run(query)
    cleaned_result = clean_ai_response(result)

    return {"answer": cleaned_result, "file_id": file_id}

# -------------------------------
# /extract-clauses: Extract clauses from uploaded PDF
# -------------------------------
@app.post("/extract-clauses")
async def extract_clauses_from_pdf(file: UploadFile = None):
    """Extract clauses from uploaded PDF file"""
    if file is None:
        return {"error": "No file uploaded."}

    try:
        # Save uploaded file temporarily
        file_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        # Initialize clause extractor
        extractor = ClauseExtractor(api_key=GEMINI_API_KEY)
        result = extractor.extract_clauses_from_pdf(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return result
    except Exception as e:
        return {"error": f"Failed to extract clauses: {str(e)}"}

# -------------------------------
# /extract-clauses-from-text: Extract clauses from text
# -------------------------------
@app.post("/extract-clauses-from-text")
async def extract_clauses_from_text(document_text: str = Form(...)):
    """Extract clauses from document text"""
    if not document_text or not document_text.strip():
        return {"error": "No text provided."}

    try:
        # Initialize clause extractor
        extractor = ClauseExtractor(api_key=GEMINI_API_KEY)
        result = extractor.extract_clauses_from_text(document_text)
        
        return result
    except Exception as e:
        return {"error": f"Failed to extract clauses from text: {str(e)}"}

# -------------------------------
# /compare-clauses: Compare clauses between two PDFs
# -------------------------------
@app.post("/compare-clauses")
async def compare_clauses(file1: UploadFile = None, file2: UploadFile = None):
    """Compare clauses between two uploaded PDF files"""
    if not file1 or not file2:
        return {"error": "Two files are required for comparison."}

    try:
        # Initialize clause extractor
        extractor = ClauseExtractor(api_key=GEMINI_API_KEY)
        
        # Process first file
        file1_bytes = await file1.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file1:
            tmp_file1.write(file1_bytes)
            tmp_file1_path = tmp_file1.name
        
        result1 = extractor.extract_clauses_from_pdf(tmp_file1_path)
        os.unlink(tmp_file1_path)
        
        if "error" in result1:
            return result1
        
        # Process second file
        file2_bytes = await file2.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file2:
            tmp_file2.write(file2_bytes)
            tmp_file2_path = tmp_file2.name
        
        result2 = extractor.extract_clauses_from_pdf(tmp_file2_path)
        os.unlink(tmp_file2_path)
        
        if "error" in result2:
            return result2
        
        # Compare clauses
        comparison = extractor.compare_clauses(
            result1.get("clauses", []),
            result2.get("clauses", [])
        )
        
        return {
            "document1": {
                "filename": file1.filename,
                "clauses": result1.get("clauses", []),
                "total_clauses": result1.get("total_clauses", 0)
            },
            "document2": {
                "filename": file2.filename,
                "clauses": result2.get("clauses", []),
                "total_clauses": result2.get("total_clauses", 0)
            },
            "comparison": comparison
        }
        
    except Exception as e:
        return {"error": f"Failed to compare clauses: {str(e)}"}
