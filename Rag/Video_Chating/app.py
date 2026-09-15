import os
import time
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate


# ============================================================
# 1. ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        f"\n❌ GEMINI_API_KEY not found.\n"
        f"Expected .env file at:\n{ENV_PATH}\n\n"
        f"Your .env should contain:\n"
        f"GEMINI_API_KEY=your_key_here"
    )


# ============================================================
# 2. EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(url: str) -> str:

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname

    if hostname:
        hostname = hostname.lower()

    # Standard YouTube URL
    if hostname in {
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    }:

        query_params = parse_qs(parsed_url.query)

        video_ids = query_params.get("v")

        if video_ids:
            return video_ids[0]

    # Short YouTube URL
    if hostname == "youtu.be":

        video_id = parsed_url.path.strip("/")

        if video_id:
            return video_id

    raise ValueError(
        "Invalid YouTube URL.\n"
        "Example:\n"
        "https://www.youtube.com/watch?v=VIDEO_ID"
    )


# ============================================================
# 3. GET YOUTUBE TRANSCRIPT
# ============================================================

def get_transcript(video_id: str) -> str:

    api = YouTubeTranscriptApi()

    try:

        transcript_data = api.fetch(
            video_id,
            languages=["en"],
        )

        transcript = " ".join(
            snippet.text
            for snippet in transcript_data
        )

        if not transcript.strip():
            raise RuntimeError(
                "Transcript is empty."
            )

        return transcript.strip()

    except TranscriptsDisabled:

        raise RuntimeError(
            "This video has captions disabled."
        )

    except NoTranscriptFound:

        raise RuntimeError(
            "No English transcript was found for this video."
        )

    except VideoUnavailable:

        raise RuntimeError(
            "This YouTube video is unavailable."
        )

    except (RequestBlocked, IpBlocked):

        raise RuntimeError(
            "YouTube blocked the transcript request. "
            "This is a YouTube IP/network restriction."
        )


# ============================================================
# 4. TEXT SPLITTING
# ============================================================

def split_transcript(transcript: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.create_documents(
        [transcript]
    )


# ============================================================
# 5. GEMINI EMBEDDINGS
# ============================================================

def create_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY,
    )


# ============================================================
# 6. RATE-LIMIT SAFE GEMINI EMBEDDING + FAISS
# ============================================================

def create_vector_store(documents):

    embeddings = create_embeddings()

    total = len(documents)

    print(f"\nTotal chunks to embed: {total}")

    # Keep batches conservative for Gemini free-tier limits.
    batch_size = 20

    all_vectors = []

    for start in range(0, total, batch_size):

        end = min(
            start + batch_size,
            total,
        )

        batch_docs = documents[start:end]

        print(
            f"\n🧠 Embedding chunks "
            f"{start + 1}-{end} of {total}..."
        )

        while True:

            try:

                vectors = embeddings.embed_documents(
                    [
                        doc.page_content
                        for doc in batch_docs
                    ]
                )

                all_vectors.extend(vectors)

                print("✅ Batch completed.")

                break

            except Exception as e:

                error_text = str(e)

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    print(
                        "\n⚠️ Gemini embedding rate limit reached."
                    )

                    print(
                        "Waiting 60 seconds before retry..."
                    )

                    time.sleep(60)

                else:

                    raise

        # Small delay between batches.
        if end < total:

            time.sleep(5)

    # Build FAISS using precomputed vectors.
    text_embeddings = [
        (
            documents[i].page_content,
            all_vectors[i],
        )
        for i in range(total)
    ]

    metadatas = [
        documents[i].metadata
        for i in range(total)
    ]

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=metadatas,
    )

    return vector_store


# ============================================================
# 7. RETRIEVER
# ============================================================

def create_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4,
        },
    )


# ============================================================
# 8. GEMINI CHAT MODEL
# ============================================================

def create_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
    )


# ============================================================
# 9. RAG PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions
about a YouTube video.

Answer ONLY using the provided transcript context.

Do not use outside knowledge.

If the answer cannot be found in the transcript,
say exactly:

"I don't know based on this video."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question",
    ],
)


# ============================================================
# 10. CLEAN GEMINI RESPONSE
# ============================================================

def extract_response_text(response) -> str:
    """
    Convert different Gemini/LangChain response formats
    into a normal Python string.

    Handles:
    - plain string
    - list of content blocks
    - dictionaries containing text
    - AIMessage-like objects
    """

    # --------------------------------------------------------
    # Case 1: Plain string
    # --------------------------------------------------------

    if isinstance(response, str):
        return response.strip()

    # --------------------------------------------------------
    # Case 2: AIMessage / response object
    # --------------------------------------------------------

    content = getattr(
        response,
        "content",
        response,
    )

    # --------------------------------------------------------
    # Case 3: content is already a string
    # --------------------------------------------------------

    if isinstance(content, str):
        return content.strip()

    # --------------------------------------------------------
    # Case 4: content is a list of blocks
    # --------------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for block in content:

            # Example:
            # {
            #   "type": "text",
            #   "text": "...",
            #   "_text": "..."
            # }

            if isinstance(block, dict):

                text = (
                    block.get("text")
                    or block.get("_text")
                    or block.get("content")
                )

                if text:
                    text_parts.append(
                        str(text)
                    )

            elif isinstance(block, str):

                text_parts.append(block)

        if text_parts:

            return "\n".join(
                text_parts
            ).strip()

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return str(content).strip()


# ============================================================
# 11. ANSWER QUESTION USING RAG
# ============================================================

def answer_question(
    retriever,
    llm,
    question: str,
):

    # Retrieve relevant transcript chunks
    retrieved_docs = retriever.invoke(
        question
    )

    # Build context
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Create final prompt
    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Ask Gemini
    response = llm.invoke(
        final_prompt
    )

    # IMPORTANT:
    # Extract ONLY readable text.
    answer = extract_response_text(
        response
    )

    return answer


# ============================================================
# 12. MAIN APPLICATION
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("             🎥 YOUTUBE VIDEO RAG CHAT")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1 - URL
    # --------------------------------------------------------

    url = input(
        "\nPaste YouTube video URL:\n> "
    ).strip()

    try:

        # ----------------------------------------------------
        # STEP 2 - VIDEO ID
        # ----------------------------------------------------

        video_id = extract_video_id(
            url
        )

        print(
            f"\n✅ Video ID: {video_id}"
        )

        # ----------------------------------------------------
        # STEP 3 - TRANSCRIPT
        # ----------------------------------------------------

        print(
            "\n📥 Fetching transcript..."
        )

        transcript = get_transcript(
            video_id
        )

        print(
            "✅ Transcript fetched successfully."
        )

        print(
            f"   Characters: {len(transcript):,}"
        )

        # ----------------------------------------------------
        # STEP 4 - SPLIT
        # ----------------------------------------------------

        print(
            "\n✂️ Splitting transcript..."
        )

        documents = split_transcript(
            transcript
        )

        print(
            f"✅ Created {len(documents)} chunks."
        )

        # ----------------------------------------------------
        # STEP 5 - EMBEDDINGS + FAISS
        # ----------------------------------------------------

        print(
            "\n🧠 Building vector database..."
        )

        vector_store = create_vector_store(
            documents
        )

        print(
            "\n✅ FAISS vector store ready."
        )

        # ----------------------------------------------------
        # STEP 6 - RETRIEVER
        # ----------------------------------------------------

        retriever = create_retriever(
            vector_store
        )

        print(
            "✅ Retriever ready."
        )

        # ----------------------------------------------------
        # STEP 7 - GEMINI
        # ----------------------------------------------------

        print(
            "\n🤖 Initializing Gemini..."
        )

        llm = create_llm()

        print(
            "✅ Gemini ready."
        )

        # ----------------------------------------------------
        # SYSTEM READY
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("              ✅ RAG SYSTEM READY")
        print("=" * 65)

        print(
            "\nAsk questions about the video."
        )

        print(
            "Type 'exit' to close."
        )

        # ----------------------------------------------------
        # CHAT LOOP
        # ----------------------------------------------------

        while True:

            question = input(
                "\nYou:\n> "
            ).strip()

            if not question:
                continue

            if question.lower() == "exit":

                print(
                    "\n👋 Goodbye!"
                )

                break

            try:

                print(
                    "\n🔎 Retrieving relevant context..."
                )

                answer = answer_question(
                    retriever,
                    llm,
                    question,
                )

                print(
                    "\n🤖 Assistant:\n"
                )

                # IMPORTANT:
                # Print only clean text.
                print(answer)

            except Exception as e:

                error_text = str(e)

                print(
                    "\n❌ Error while answering:"
                )

                print(error_text)

    except Exception as e:

        print(
            "\n❌ Application failed:"
        )

        print(str(e))


# ============================================================
# 13. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()