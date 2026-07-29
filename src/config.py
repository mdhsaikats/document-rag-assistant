import os
from dotenv import load_dotenv

load_dotenv()

def get_google_key():
    return os.getenv("GOOGLE_API_KEY")