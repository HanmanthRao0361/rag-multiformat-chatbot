import streamlit as st
import os

# Import our classes
from data_loader import DataLoader
from rag_chain import RAGChain

# Page settings
st.set_page_config(page_title="RAG Chat", layout="wide")

# Title
st.title("📚 Multi-Format Document Chat")
st.markdown("*Upload PDF, Excel, Word, TXT files*")

# Sidebar - show status
st.sidebar.header("Status")
if os.path.exists("faiss_index"):
    st.sidebar.success("✅ Ready to chat!")
else:
    st.sidebar.warning("⏳ Process documents first")

# === STEP 1: Upload and Process ===
st.header("Step 1: Upload Documents")
col1, col2 = st.columns([4, 1])

with col1:
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['pdf', 'xlsx', 'xls', 'docx', 'txt'],
        accept_multiple_files=True
    )

with col2:
    # Process button
    if st.button("Process Files", type="primary"):
        if uploaded_files:
            with st.spinner("Processing files..."):
                # Create data loader
                loader = DataLoader()
                
                # Load all documents
                docs = loader.load_documents(uploaded_files)
                
                # Create vector database
                loader.create_vectorstore(docs)
                
                st.success("✅ Documents processed!")
                st.rerun()
        else:
            st.error("Upload files first!")

# Show uploaded files
if uploaded_files:
    for file in uploaded_files:
        st.info(f"📄 {file.name}")

# === STEP 2: Chat ===
st.header("Step 2: Ask Questions")
if os.path.exists("faiss_index"):
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # New question input
    question = st.chat_input("Ask about your documents...")
    
    if question:
        # Add user question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Getting answer..."):
                rag = RAGChain()
                answer = rag.answer_question(question)
                st.write(answer)
                
                # Save answer
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👆 **Upload files and click Process first**")
