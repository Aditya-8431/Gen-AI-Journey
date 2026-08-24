from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_output_tokens=300
)

parser = StrOutputParser()



prompt1 = PromptTemplate(
    template="""
Classify the following customer feedback as either
Positive or Negative.

Feedback:
{feedback}

Return only one word: Positive or Negative.
""",
    input_variables=["feedback"]
)

classifier_chain = prompt1 | model | parser




prompt2 = PromptTemplate(
    template="""
Write an appropriate thank-you response for this
positive customer feedback:

{feedback}
""",
    input_variables=["feedback"]
)

positive_chain = prompt2 | model | parser



prompt3 = PromptTemplate(
    template="""
Write an appropriate apology response for this
negative customer feedback:

{feedback}
""",
    input_variables=["feedback"]
)

negative_chain = prompt3 | model | parser



branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].lower() == "positive",
        positive_chain
    ),
    (
        lambda x: x["sentiment"].lower() == "negative",
        negative_chain
    ),
    RunnableLambda(
        lambda x: "Could not determine the sentiment."
    )
)