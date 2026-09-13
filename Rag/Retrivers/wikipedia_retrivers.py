import wikipedia

from langchain_community.retrievers import WikipediaRetriever


# Set user agent
wikipedia.set_user_agent("LangChain-Retriever/1.0")

# Create retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

# Search query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Get documents
docs = retriever.invoke(query)

# Print results
for i, doc in enumerate(docs):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)