from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

massages=[
    SystemMessage(content="You are a helpful assistant ."),
    HumanMessage(content="Tell me about Python programming language."),
]
result=model.invoke(massages)

massages.append(AIMessage(content=result.text))

print("AI: ",massages)