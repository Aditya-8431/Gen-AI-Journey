import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


# Load API key
load_dotenv()

# Create documents
documents = [
    Document(
        page_content="LangChain helps developers build LLM applications easily."
    ),
    Document(
        page_content="Chroma is a vector database optimized for LLM-based search."
    ),
    Document(
        page_content="Embeddings convert text into high-dimensional vectors."
    ),
    Document(
        page_content="OpenAI provides powerful embedding models."
    ),
]

# Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Search query
query = "What is Chroma used for?"
results = retriever.invoke(query)

# Print results
for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)