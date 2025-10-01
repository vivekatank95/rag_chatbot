import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
EMBED_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

def get_retriever(k=4):
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever

# In src/retriever.py
def retrieve_docs(query: str, k=6):
    retriever = get_retriever(k)
    docs = retriever.get_relevant_documents(query)
    # Optional: Add working print statements here for debugging if needed
    # print(f"Retrieved {len(docs)} documents for query: {query}")
    return docs