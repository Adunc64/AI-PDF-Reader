import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from vector_store import create_vector_store
from pdf_loader import load_pdf, split_text
from qa_agent import ask_question

st.set_page_config(page_title="PDF AI Agent", layout="wide")

st.title("📄 AI PDF Reader & Analyzer")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")
    st.warning("Uploading a new PDF will replace previous data.")

    # Process PDF
    docs = load_pdf("data/temp.pdf")
    chunks = split_text(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    create_vector_store(chunks, embedding_model)

    st.info("PDF processed and ready for questions!")

# Chat section
st.subheader("Ask Questions")

query = st.text_input("Enter your question:")

if query:
    answer = ask_question(query)
    st.write("### 🤖 Answer:")
    st.write(answer)