

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Define model and chain
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash")
chain = model | StrOutputParser()

# 2. Invoke the CHAIN (not the model)
result = chain.invoke("Genrate a image of tree")

# 3. Print the clean text directly
print(result)