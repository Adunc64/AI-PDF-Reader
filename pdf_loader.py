from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#load file
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

#split into chunks
def split_text(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    file_path = "data/sample.pdf"

    docs = load_pdf(file_path)
    chunks = split_text(docs)

    print(f"Total pages loaded: {len(docs)}")
    print(f"Total chunks created: {len(chunks)}")

    print("\nSample chunk:\n")
    print(chunks[0].page_content)