import os
from src.config import get_openai_key
from src.loader import load_and_chunk_pdf
from src.database import create_or_load_database
from src.generator import answer_question

def main():
    # Ensure API key is loaded
    if not get_openai_key():
        print("Error: OPENAI_API_KEY not found in .env file.")
        return

    # Check if the database already exists to avoid rebuilding it every time
    if not os.path.exists("./chroma_db"):
        # STEP 1: Load and chunk data
        pdf_path = "./data/sample.pdf" 
        chunks = load_and_chunk_pdf(pdf_path)
        
        # STEP 2: Build and save the database
        vector_db = create_or_load_database(chunks)
    else:
        # Just load the database if we already built it previously
        vector_db = create_or_load_database()

    # STEP 3: Ask a question!
    print("\n--- System Ready ---")
    question = "What is the main topic of this document?"
    print(f"Question: {question}")
    
    answer = answer_question(vector_db, question)
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()