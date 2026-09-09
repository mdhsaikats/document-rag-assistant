from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./chroma_db"


def create_or_load_database(chunks=None):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

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