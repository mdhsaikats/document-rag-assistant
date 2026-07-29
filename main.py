import os
from src.config import get_google_key
from src.loader import load_and_chunk_pdf
from src.database import create_or_load_database
from src.generator import answer_question

def main():
    # Check for Google Key
    if not get_google_key():
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    # Check if the database already exists
    if not os.path.exists("./chroma_db"):
        print("Starting fresh. Loading PDF...")
        pdf_path = "./data/sample.pdf" 
        chunks = load_and_chunk_pdf(pdf_path)
        vector_db = create_or_load_database(chunks)
    else:
        vector_db = create_or_load_database()

    # Ask the question
    print("\n--- System Ready ---")
    question = "What is the main topic of this document?"
    print(f"Question: {question}")
    
    answer = answer_question(vector_db, question)
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()