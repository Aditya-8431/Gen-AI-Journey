import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


# Load API key
load_dotenv()

# Create documents
docs = [
    Document(
        page_content="LangChain makes it easy to work with LLMs."
    ),
    Document(
        page_content="LangChain is used to build LLM based applications."
    ),
    Document(
        page_content="Chroma is used to store and search document embeddings."
    ),
    Document(
        page_content="Embeddings are vector representations of text."
    ),
    Document(
        page_content="MMR helps you get diverse results when doing similarity search."
    ),
    Document(
        page_content="LangChain supports Chroma, FAISS, Pinecone, and more."
    ),
]

# Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Create vector store
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Create MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5
    }
)

# Search query
query = "What is langchain?"
results = retriever.invoke(query)

# Print results
for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)