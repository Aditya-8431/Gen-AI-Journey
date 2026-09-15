import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings
)


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(
    PROJECT_ROOT / ".env"
)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found."
    )


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

vector = embeddings.embed_query(
    "What is Retrieval Augmented Generation?"
)

print("✅ Gemini embedding works.")
print("Vector size:", len(vector))