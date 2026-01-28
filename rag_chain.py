# Load API key from .env file
import os
from dotenv import load_dotenv

# Load Groq model
from langchain_groq import ChatGroq

# Load vector database
from langchain_community.vectorstores import FAISS

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# ⭐ YOUR SYSTEM PROMPT
from langchain_core.messages import SystemMessage

load_dotenv()

class RAGChain:
    def __init__(self):
        # Create embeddings
        self.embeddings = HuggingFaceEmbeddings()
        
        # Create Groq model (UPDATED MODEL)
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant"  
        )
        
        #  YOUR SYSTEM PROMPT - Data analysis rules
        self.system_prompt = SystemMessage(content="""
You are a data analysis assistant.

When calculating averages:
- Use only the explicitly mentioned column.
- Do NOT use Total_Amount unless asked.
- Do NOT weight by Quantity unless asked.
- Treat each row as one independent record.
- Do not assume business logic that is not stated.
""")
    
    def load_vectorstore(self):
        # Check if vector database exists
        if not os.path.exists("faiss_index"):
            print("No vector database found!")
            return None
        
        # Load vector database
        vectorstore = FAISS.load_local(
            "faiss_index", self.embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    
    def answer_question(self, question):
        # Load vector database
        vectorstore = self.load_vectorstore()
        if vectorstore is None:
            return "Please process documents first!"
        
        # Find 3 most relevant document pieces
        relevant_docs = vectorstore.similarity_search(question, k=3)
        
        # Combine document content
        context = ""
        for doc in relevant_docs:
            context += doc.page_content + "\n\n"
        
        # ⭐ USE SYSTEM PROMPT + Context + Question
        full_prompt = [
            self.system_prompt,  # YOUR RULES
            ("human", f"""Use only the context below to answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")
        ]
        
        # Get answer from Groq WITH SYSTEM PROMPT
        response = self.llm.invoke(full_prompt)
        return response.content
