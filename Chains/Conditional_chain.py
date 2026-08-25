from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda
)
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_output_tokens=300
)

parser = StrOutputParser()


# Classifier
prompt1 = PromptTemplate(
    template="""
Classify this customer feedback as Positive or Negative.

Feedback:
{feedback}

Return only Positive or Negative.
""",
    input_variables=["feedback"]
)

classifier_chain = prompt1 | model | parser


# Positive
prompt2 = PromptTemplate(
    template="""
Write a short thank-you response for this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

positive_chain = prompt2 | model | parser


# Negative
prompt3 = PromptTemplate(
    template="""
Write a short apology response for this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

negative_chain = prompt3 | model | parser


# Conditional Chain
def conditional_chain(inputs):

    sentiment = classifier_chain.invoke({
        "feedback": inputs["feedback"]
    })

    if sentiment.strip().lower() == "positive":
        return positive_chain.invoke({
            "feedback": inputs["feedback"]
        })

    elif sentiment.strip().lower() == "negative":
        return negative_chain.invoke({
            "feedback": inputs["feedback"]
        })

    else:
        return "Could not determine the sentiment."


result = conditional_chain({
    "feedback": "This is a wonderful smartphone!"
})

print(result)