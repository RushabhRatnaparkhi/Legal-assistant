import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


class QueryAgent:
    """Handles retrieving legal information from PDFs."""
    
    def __init__(self, pdf_path, name, api_key):
        self.pdf_path = pdf_path
        self.name = name

        # Validate API Key passed explicitly
        if not api_key:
            raise ValueError("❌ Google Gemini API key is missing!")

        # Create embeddings object using provided key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Build vectorstore
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever()

    def _create_vectorstore(self):
        """Loads PDF, splits text, and stores it in FAISS."""
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def get_relevant_text(self, query):
        """Retrieve relevant sections from PDFs."""
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs]) if docs else None
