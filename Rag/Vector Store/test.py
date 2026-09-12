import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

print("API key loaded:", bool(os.getenv("GEMINI_API_KEY")))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

print("Gemini + Chroma initialized successfully!")