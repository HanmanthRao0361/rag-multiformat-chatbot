# Simple class to load PDF, Excel, Word, TXT files
import os
import pandas as pd  # For Excel files

# Loaders for different files
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredWordDocumentLoader, 
    TextLoader
)

# ⭐ FIXED: Document class moved to langchain_core
from langchain_core.documents import Document

# Split long text into small pieces
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Convert text to numbers (embeddings)
from langchain_huggingface import HuggingFaceEmbeddings

# Save embeddings in database
from langchain_community.vectorstores import FAISS

class DataLoader:
    def __init__(self):
        # Create embeddings model (text → numbers)
        self.embeddings = HuggingFaceEmbeddings()
        
        # Split big text into 1000 char pieces
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
    
    def load_documents(self, uploaded_files):
        # List to store all documents
        all_docs = []
        
        # Loop through each uploaded file
        for file in uploaded_files:
            if file is None:
                continue
                
            # Save uploaded file to temp location
            temp_file = f"./temp_{file.name}"
            with open(temp_file, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                print(f"Loading file: {file.name}")
                
                # ⭐ EXCEL FILES - SIMPLE PANDAS METHOD (WORKS EVERY TIME!)
                if file.name.endswith(('.xlsx', '.xls')):
                    # Read Excel with pandas
                    df = pd.read_excel(temp_file)
                    excel_text = df.to_string()
                    
                    # Create document from Excel data
                    excel_doc = Document(
                        page_content=excel_text, 
                        metadata={"source": file.name, "type": "excel"}
                    )
                    all_docs.append(excel_doc)
                    print(f"✅ Excel loaded: {file.name}")
                
                # PDF files
                elif file.name.endswith('.pdf'):
                    loader = PyPDFLoader(temp_file)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"✅ PDF loaded: {file.name}")
                
                # Word files
                elif file.name.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(temp_file)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"✅ Word loaded: {file.name}")
                
                # Text files
                elif file.name.endswith('.txt'):
                    loader = TextLoader(temp_file)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"✅ Text loaded: {file.name}")
                
                else:
                    print(f"❌ Skip unsupported file: {file.name}")
            
            except Exception as e:
                print(f"❌ Error loading {file.name}: {str(e)}")
            
            finally:
                # Always delete temp file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        
        print(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def create_vectorstore(self, documents):
        # Check if we have documents
        if not documents:
            print("❌ No documents to process!")
            return None
        
        # Split documents into small chunks
        print("Splitting text into chunks...")
        text_chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(text_chunks)} text chunks")
        
        # Create vector database
        print("Creating vector database...")
        vector_db = FAISS.from_documents(text_chunks, self.embeddings)
        
        # Save to folder
        vector_db.save_local("faiss_index")
        print("✅ Vector database saved as 'faiss_index'!")
        return vector_db
