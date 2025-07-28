#STREAMLIT APP FOR TESTING


# import os
# import streamlit as st
# import tempfile
# from utils.query_agent import QueryAgent
# from utils.summarization import SummarizationAgent
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA

# # Load API Key securely
# API_KEY = st.secrets.get("GEMINI_API_KEY")
# if not API_KEY:
#     st.error("❌ GEMINI_API_KEY not found in secrets.toml")
#     st.stop()


# st.set_page_config(page_title="Legal Chatbot", layout="wide")
# st.title("⚖️ Legal AI Chatbot")
# st.write("Ask legal questions related to **Indian Litigation & Corporate Laws**.")


# pdf_paths = {
#     "Guide to Litigation in India": "data/Guide-to-Litigation-in-India.pdf",
#     "Legal Compliance & Corporate Laws": "data/Legal-Compliance-Corporate-Laws.pdf",
#     "legaldoc":"data/legaldoc.pdf",
#     "IPC":"data/penal_code.pdf",
#     "Constitution of India": "data/constitution_of_india.pdf",
# }

# # Sidebar selection
# option = st.sidebar.radio(
#     "Choose PDF source",
#     ("Ask from our existing Legal Database", "Ask from your own PDF")
# )

# # Option 1: Existing PDFs
# if option == "Ask from our existing Legal Database":
#     class LegalChatbot:
#         def __init__(self, pdf_paths):
#             self.query_agents = {
#                 name: QueryAgent(path, name, API_KEY)  # 👈 Pass API_KEY to QueryAgent
#                 for name, path in pdf_paths.items()
#             }
#             self.summarizer = SummarizationAgent()

#         def answer_query(self, query):
#             best_answer = None
#             best_source = None
#             best_score = 0
#             for name, agent in self.query_agents.items():
#                 retrieved_text = agent.get_relevant_text(query)
#                 if retrieved_text:
#                     summary = self.summarizer.summarize(retrieved_text)
#                     score = len(retrieved_text)
#                     if score > best_score:
#                         best_score = score
#                         best_answer = summary
#                         best_source = name
#             if best_answer:
#                 return f"""📖 **Source:** {best_source}\n\n💡 **Answer:**\n{best_answer}"""
#             else:
#                 return "❌ No relevant legal information found."

#     bot = LegalChatbot(pdf_paths)
#     query = st.text_input("📝 Enter your legal question:")
#     if st.button("Ask"):
#         if query:
#             with st.spinner("🔍 Searching..."):
#                 response = bot.answer_query(query)
#                 st.write(response)
#         else:
#             st.warning("⚠️ Please enter a question!")

# # Option 2: Upload Your Own PDF
# else:
#     uploaded_pdf = st.file_uploader("📤 Upload your legal PDF", type="pdf")
#     query = st.text_input("📝 Enter your legal question:")

#     if uploaded_pdf and query:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_pdf.read())
#             tmp_path = tmp_file.name

#         # Load and embed
#         loader = PyPDFLoader(tmp_path)
#         docs = loader.load()
#         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = splitter.split_documents(docs)

#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=API_KEY  # ✅ Pass key here
#         )

#         temp_vectorstore = FAISS.from_documents(chunks, embeddings)

#         def qa_from_vectorstore(vectorstore, query):
#             llm = ChatGoogleGenerativeAI(
#                 model="models/gemini-2.5-pro",
#                 temperature=0.3,
#                 google_api_key=API_KEY  # ✅ Pass key here
#             )
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 retriever=vectorstore.as_retriever()
#             )
#             return qa_chain.run(query)

#         with st.spinner("🔍 Searching in your PDF..."):
#             response = qa_from_vectorstore(temp_vectorstore, query)
#             st.markdown(f"**Answer:** {response}")

#     elif not uploaded_pdf and query:
#         st.warning("⚠️ Please upload a PDF.")
