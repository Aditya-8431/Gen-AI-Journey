from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_output_tokens=2000
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "cricket"})

print(result)

# See the chain structure
chain.get_graph().print_ascii()