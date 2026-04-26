from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from pdf_loader import load_pdf, split_text

#create embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#create and store vector database
def create_vector_store(chunks, embedding_model):
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="db")
    vectorstore.persist()
    return vectorstore

if __name__ == "__main__":
    file_path = "data/sample.pdf"

    docs = load_pdf(file_path)
    chunks = split_text(docs)

    vectorstore = create_vector_store(chunks, embedding_model)

    print("Vector store created and persisted successfully.")
