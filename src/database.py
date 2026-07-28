from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os

DB_DIR = "./chroma_db"

def create_or_load_database(chunks=None):
    embeddings = OpenAIEmbeddings()

    if chunks:
        print("Building the database...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
    else:
        print("Loading existing database...")
        vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    return vector_db