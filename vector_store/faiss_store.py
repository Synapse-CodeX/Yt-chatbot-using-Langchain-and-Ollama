from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

