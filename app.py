import streamlit as st
import os
from config import GROQ_API_KEY, PINECONE_API_KEY, DEFAULT_INDEX_NAME, DEFAULT_NAMESPACE
from src.pdf_processor import process_pdf
from src.vector_store import index_to_pinecone
from src.rag_chain import generate_answer

st.set_page_config(page_title="Intermediate RAG System", page_icon="📘", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_doc" not in st.session_state:
    st.session_state.processed_doc = False

st.sidebar.title("🛠️ Configuration & Settings")
groq_key = st.sidebar.text_input("Groq API Key", type="password", value=GROQ_API_KEY)
pinecone_key = st.sidebar.text_input("Pinecone API Key", type="password", value=PINECONE_API_KEY)
index_name = st.sidebar.text_input("Pinecone Index Name", value=DEFAULT_INDEX_NAME)
namespace = st.sidebar.text_input("Pinecone Namespace", value=DEFAULT_NAMESPACE)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Hyperparameters")
chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, 500, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 50, 10)
top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, 4)
similarity_threshold = st.sidebar.slider("Similarity Threshold Score", 0.0, 1.0, 0.3, 0.05)
enable_ocr = st.sidebar.checkbox("Enable OCR Fallback", value=True)

st.title("📘 Intermediate RAG System with Pinecone")
st.caption("Ask questions strictly based on your uploaded PDF document.")

uploaded_file = st.file_uploader("Upload PDF Document (Max 20MB)", type=["pdf"])

if uploaded_file:
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("❌ File size exceeds the 20MB limit.")
    elif st.button("🚀 Process & Index Document"):
        if not groq_key or not pinecone_key:
            st.error("🔑 Please enter API Keys in sidebar!")
        else:
            with st.spinner("Processing PDF & indexing to Pinecone..."):
                try:
                    chunks = process_pdf(uploaded_file, chunk_size, chunk_overlap, enable_ocr)
                    st.session_state.vector_store = index_to_pinecone(chunks, pinecone_key, index_name, namespace)
                    st.session_state.processed_doc = True
                    st.success(f"✅ Success! Processed {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Retrieved Sources"):
                for src in msg["sources"]:
                    st.write(f"**Page:** {src['page']} | **Score:** {src['score']:.4f}")
                    st.caption(f"\"{src['text']}\"")

if query := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.processed_doc or not st.session_state.vector_store:
        st.warning("⚠️ Upload and process a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching & generating answer..."):
                answer, sources = generate_answer(
                    query, st.session_state.vector_store, groq_key, top_k, similarity_threshold, namespace
                )
                st.write(answer)
                if sources:
                    with st.expander("📚 View Retrieved Sources"):
                        for src in sources:
                            st.write(f"**Page:** {src['page']} | **Score:** {src['score']:.4f}")
                            st.caption(f"\"{src['text']}\"")

                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})