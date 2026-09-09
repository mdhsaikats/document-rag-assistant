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

    # Ask questions interactively
    print("\n--- System Ready ---")
    print("Type your questions below. Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            question = input("\nQuestion: ")
            if question.strip().lower() in ["exit", "quit"]:
                print("Exiting RAG system...")
                break
            if not question.strip():
                continue
                
            answer = answer_question(vector_db, question)
            print(f"Answer: {answer}")
        except KeyboardInterrupt:
            print("\nExiting RAG system...")
            break
        except Exception as e:
            print(f"Error answering question: {e}")

if __name__ == "__main__":
    main()