from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash" 
)
chat_history = [
        SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input= input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input== "exit":
        break
    result= model.invoke(user_input)
    chat_history.append(AIMessage(content=result.text))
  #  print("AI: ", result.content)
    print("Chat History: ", chat_history)