import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from utils.query_agent import QueryAgent
from utils.summarization import SummarizationAgent
from dotenv import load_dotenv
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]


@app.post("/ask-existing")
async def ask_from_existing(query: str = Form(...)):
    pdf_paths = {
        "Guide to Litigation in India": "data/Guide-to-Litigation-in-India.pdf",
        "Legal Compliance & Corporate Laws": "data/Legal-Compliance-Corporate-Laws.pdf",
        "legaldoc":"data/legaldoc.pdf",
        "IPC":"data/penal_code.pdf",
        "Constitution of India": "data/constitution_of_india.pdf",

    }

    summarizer = SummarizationAgent()
    best_answer = None
    best_score = 0
    best_source = None

    for name, path in pdf_paths.items():
        agent = QueryAgent(path, name, GEMINI_API_KEY)
        result = agent.get_relevant_text(query)
        if result:
            summary = summarizer.summarize(result)
            score = len(result)
            if score > best_score:
                best_score = score
                best_answer = summary
                best_source = name

    if best_answer:
        return {"source": best_source, "answer": best_answer}
    else:
        return {"error": "No relevant information found"}


@app.post("/ask-upload")
async def ask_from_uploaded(query: str = Form(...), file: UploadFile = None):
    if file is None:
        return {"error": "No PDF file uploaded"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    result = qa_chain.run(query)
    return {"answer": result}
