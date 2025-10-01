import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use token for embeddings / retriever


# Config
DATA_DIR = "./data"          # folder where your PDFs/DOCX are stored
PERSIST_DIR = "./chroma_db"  # folder where Chroma DB will be saved
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(data_dir: str) -> List:
    docs = []
    p = Path(data_dir)
    for f in p.glob("**/*"):
        if f.is_file() and f.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(f))
            docs.extend(loader.load())
    if not docs:
        raise ValueError("No PDF documents found in the directory.")
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_and_persist(chunks):
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedder, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"Persisted vectorstore to {PERSIST_DIR}")

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    embed_and_persist(chunks)
