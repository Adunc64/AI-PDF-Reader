from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# load embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# load vectorstore
def load_vectorstore():
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory="db"
    )
    return vectorstore

# create retriever
def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

# load llm
def get_llm():
    llm = OllamaLLM(model="llama3")
    return llm

# Ask question
def ask_question(query):
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    llm = get_llm()

    # ✅ FIXED LINE
    relevant_docs = retriever.invoke(query)

    # combine retrieved chunks into context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # prompt
    prompt = f"""You are an AI assistant. Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:
"""

    # generate answer using llm
    answer = llm.invoke(prompt)
    return answer


if __name__ == "__main__":
    print("📄 PDF Chat Agent Ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        answer = ask_question(query)
        print("\nAI:", answer, "\n")