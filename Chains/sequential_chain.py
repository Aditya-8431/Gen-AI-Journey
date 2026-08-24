from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model= GoogleGenerativeAI( model="gemini-3.6-flash" )

prompt=PromptTemplate(
    template="Give me a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Genrate the five points of Summery from this \n  {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

chain=prompt | model | parser | prompt2| model |parser

result=chain.invoke({ 'topic':'Using of Mobile phone as Student'})

print(result)

chain.get_graph().print_ascii()

