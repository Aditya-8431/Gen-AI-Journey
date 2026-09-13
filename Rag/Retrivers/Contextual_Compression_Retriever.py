import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor
)
from langchain_core.documents import Document


# Load API key
load_dotenv()

# Create documents
docs = [
    Document(
        page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world.
            Photosynthesis is the process by which green plants convert sunlight into energy.
            Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ),
        metadata={"source": "Doc1"}
    ),

    Document(
        page_content=(
            """In medieval Europe, castles were built primarily for defense.
            The chlorophyll in plant cells captures sunlight during photosynthesis.
            Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ),
        metadata={"source": "Doc2"}
    ),

    Document(
        page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century.
            It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
        ),
        metadata={"source": "Doc3"}
    ),

    Document(
        page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form.
            Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
            Modern filmmaking involves complex CGI and sound design."""
        ),
        metadata={"source": "Doc4"}
    )
]

# Create embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Create vector store
vectorstore = FAISS.from_documents(
    docs,
    embedding_model
)

# Create base retriever
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Create compressor
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Search query
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

# Print results
for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)