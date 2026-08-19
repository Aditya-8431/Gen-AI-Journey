from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI(model='gpt-4',temperature=0.5,max_completion_token=10)
#temparature=0.5 = The temperature parameter controls the randomness of the model's output.
#  A lower temperature (closer to 0) will make the model's responses more deterministic and focused,
#  while a higher temperature (closer to 1) will make the responses more diverse and creative.
# Max_completion_token=10 = The max_completion_token parameter sets the maximum number of tokens
#  (words or word pieces) that the model can generate in its response.
result=model.invoke("What is the capital of France?")
print(result)

# This code only be when Api key is set in the environment variables. 
# Make sure to set your OpenAI API key in the .env file or your system environment variables before running this code.