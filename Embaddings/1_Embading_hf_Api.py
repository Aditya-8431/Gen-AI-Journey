import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = "sentence-transformers/all-MiniLM-L6-v2"

questions = {
    1: "What is the capital of India?",
    2: "What is machine learning?",
    3: "How does a neural network work?",
    4: "What is the difference between AI and ML?"
}

embeddings = {}

for question_id, question in questions.items():
    embedding = client.feature_extraction(
        question,
        model=model
    )

    embeddings[question_id] = embedding

for question_id, embedding in embeddings.items():
    print(f"Question {question_id}:")
    print(questions[question_id])
    print("Embedding dimension:", len(embedding))
    print("Embedding:", embedding)
    print("-" * 50)