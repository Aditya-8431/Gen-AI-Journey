# from langchain_text_splitters import CharacterTextSplitter

# text ="""
# Intelligent agents can range from simple to highly complex. A basic thermostat or control system is considered an intelligent agent, as is a human being, or any other system that meets the same criteria—such as a firm, a state, or a biome.[1]

# Intelligent agents operate based on an objective function, which encapsulates their goals. They are designed to create and execute plans that maximize the expected value of this function upon completion.[2] For example, a reinforcement learning agent has a reward function, which allows programmers to shape its desired behavior.[3] Similarly, an evolutionary algorithm's behavior is guided by a fitness function.[4]

# Intelligent agents in artificial intelligence are closely related to agents in economics, and versions of the intelligent agent paradigm are studied in cognitive science, ethics, and the philosophy of practical reason, as well as in many interdisciplinary socio-cognitive modeling and computer social simulations.
# """
# spliter= CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0,
#     separator=" "

# )
# chunks=spliter.split_text(text)
# print(chunks)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter




# Update this path if the PDF is stored elsewhere
loader = PyPDFLoader(r"C:\Users\neela\OneDrive\Desktop\LangChain\Rag\Document_Loader\Accenture_technical_important.pdf")

docs=loader.load()

spliter= CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "

)
chunks=spliter.split_documents(docs)

print(chunks[0].page_content)

