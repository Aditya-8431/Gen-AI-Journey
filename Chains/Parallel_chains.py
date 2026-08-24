from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1= GoogleGenerativeAI( model="gemini-3.6-flash" )

llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    max_new_tokens=1000,
    temperature=0.7
)
model2=ChatHuggingFace(llm=llm)



prompt1=PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']

)
prompt2=PromptTemplate(
    template="Generate  5 short  Quiz  from the following text \n {text}",
    input_variables=['text']

)
prompt3=PromptTemplate(
    template="Merge the provided Notes and quiz into a single Document /n notes -> {notes} and quiz -> {quiz}",

    input_variables=['notes','quiz']

)
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1 |model1| parser,
    'quiz': prompt2| model2 | parser

})

merge_chain= prompt3 | model1| parser

chain= parallel_chain|merge_chain

text ="""
    Retrieval-Augmented Generation (RAG) is a way to make AI answers more reliable by combining searching for relevant information and then generating a response. Instead of guessing based only on old training data, it first finds useful data from external sources (like documents or databases) and then uses it to give a better answer.

Fetches up-to-date data and reduces incorrect or made-up answers
Works well with specialized data like medical or legal content
No need to retrain the model every time new data comes in
Can use user-specific data to give more relevant responses
What-is-RAG_
Retrieval Augmented Generation (RAG)
For example, a platform like GeeksforGeeks has its own large collection of coding articles and tutorials. If a user asks a question, instead of giving a general answer, a RAG-based system can:

Search relevant articles from their dataset
Pick the most useful content
Generate an answer based on that specific information
This way, the response is more accurate, aligned with the platform’s content and actually helpful for the user.

Components of RAG
The main components of RAG are:

External Knowledge Source: Stores domain specific or general information like documents, APIs or databases.
Text Chunking and Preprocessing: Breaks large text into smaller, manageable chunks and cleans it for consistency.
Embedding Model: Converts text into numerical vectors that capture semantic meaning.
Vector Database: Stores embeddings and enables similarity search for fast information retrieval.
Query Encoder: Transforms the users query into a vector for comparison with stored embeddings.
Retriever: Finds and returns the most relevant chunks from the database based on query similarity.
Prompt Augmentation Layer: Combines retrieved chunks with the user’s query to provide context to the LLM.
LLM (Generator): Generates a grounded response using both the query and retrieved knowledge.
Updater (Optional): Regularly refreshes and re-embeds data to keep the knowledge base up to date.
Working of RAG
The system first searches external sources for relevant information based on the user’s query instead of relying only on existing training data.

How-Rag-works
Training
Creating External Data: External data from APIs, databases or documents is chunked, converted into embeddings and stored in a vector database to build a knowledge library.
Retrieving Relevant Information: User queries are converted into vectors and matched against stored embeddings to fetch the most relevant data ensuring accurate responses.
Augmenting the LLM Prompt: Retrieved content is added to the user’s query giving the LLM extra context to work with.
Answer Generation: LLM uses both the query and retrieved data to generate a factually accurate, context aware response.
Keeping Data Updated: External data and embeddings are refreshed regularly in real time or scheduled so the system always retrieves latest information.
What Problems does RAG solve
Hallucinations: Traditional generative models can produce incorrect information. RAG reduces this risk by retrieving verified, external data to ground responses in factual knowledge.
Outdated Information: Static models rely on training data that may become outdated. It dynamically retrieves latest information ensuring relevance and accuracy in real time.
Contextual Relevance: Generative models often struggle with maintaining context in complex or multi turn conversations. RAG retrieves relevant documents to enrich the context improving coherence and relevance.
Domain Specific Knowledge: Generic models may lack expertise in specialized fields. It integrates domain specific external knowledge for tailored and precise responses.
Cost and Efficiency: Fine tuning large models for specific tasks is expensive. It eliminates the need for retraining by dynamically retrieving relevant data reducing costs and computational load.
Scalability Across Domains: It is adaptable to diverse industries from healthcare to finance without extensive retraining making it highly scalable.
Challenges
Complexity: Combining retrieval and generation adds complexity to the model requires careful tuning and optimization to ensure both components work seamlessly together.
Latency: The retrieval step can introduce latency making it challenging to deploy RAG models in real time applications.
Quality of Retrieval: The overall performance heavily depends on the quality of the retrieved documents. Poor retrieval can lead to suboptimal generation, undermining the model’s effectiveness.
Bias and Fairness: It can inherit biases present in the training data or retrieved documents, necessitating ongoing efforts to ensure fairness and mitigate biases.
"""

result= chain.invoke({'text':text})
print(result)
chain.get_graph().print_ascii()
