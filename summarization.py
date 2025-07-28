import os
from langchain_google_genai import ChatGoogleGenerativeAI

API_KEY = os.getenv("GEMINI_API_KEY")

class SummarizationAgent:
    """Summarizes legal information retrieved from PDFs."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini-2.5-pro", temperature=0.3)

    def summarize(self, text):
        """Simplifies legal content into user-friendly answers."""
        if not text:
            return "❌ No relevant legal information found."
        
        prompt = f"Summarize the following legal text in simple terms:\n\n{text}"
        response = self.llm.invoke(prompt)

        if isinstance(response, str):
            return response  # Already clean
        
        elif isinstance(response, dict) and "content" in response:
            return response["content"]  # ✅ Extract clean answer
        
        elif hasattr(response, "content"):
            return response.content  # ✅ Extract only the answer
        
        return "❌ No valid response received from the model."