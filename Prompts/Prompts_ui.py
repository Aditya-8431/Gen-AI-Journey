from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.header("Research Assistant")




if st.button("Submit"):
    # 1. Define model and chain
    model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    chain = model | StrOutputParser()

    # 2. Invoke the CHAIN (not the model)
    result = chain.invoke(user_input)

    # 3. Print the clean text directly
    st.write(result)