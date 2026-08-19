from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model='claude-v1',temperature=0.5,max_completion_token=10)

result=model.invoke("What is the capital of France?")
print(result)