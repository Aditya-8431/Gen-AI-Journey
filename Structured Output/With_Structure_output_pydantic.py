from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    max_output_tokens=500
)


class Review(BaseModel):

    summary: str = Field(
        description="A concise summary of the resume."
    )

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment of the resume."
    )

    Name: str = Field(
        description="Extract the person's name from the resume."
    )

    Skills: Optional[list[str]] = Field(
        default=None,
        description="List the technical skills mentioned in the resume."
    )

    Project: list[str] = Field(
        description="List each project with only the project name and technology stack."
    )


structured_output = model.with_structured_output(Review)


result = structured_output.invoke("""
This is Aditya Raj resume.

B.Tech Computer Science Engineering student with strong foundations in Java,
Python, Data Structures & Algorithms, Object-Oriented Programming, and SQL,
with 500+ DSA problems solved across multiple coding platforms.

Developed software applications using Flask, REST APIs, Machine Learning,
and Generative AI, including a research-based AI tool built using the Gemini API.

Currently exploring Generative AI fundamentals and interested in building
practical AI-powered applications, scalable software systems, and solving
real-world engineering problems.

EDUCATION:
Galgotias University — B.Tech in Computer Science Engineering
Oct 2023–Oct 2027
CGPA: 8.1/10

TECHNICAL SKILLS:
Languages: Java, Python, SQL
AI / Machine Learning: Machine Learning, NLP, Generative AI, TF-IDF,
Random Forest, Linear Regression
Backend: Flask, REST APIs
Databases: MySQL, SQLite
Core CS: Data Structures & Algorithms, OOP, Operating Systems,
Computer Networks, System Design Basics
Developer Tools: Git, GitHub, VS Code

PROJECTS:

AI-Powered Smart Expense Tracker
Python, Flask, SQLite, Machine Learning

Built an AI-powered expense management application using Flask, SQLite,
and machine learning for automated expense categorization and spending analysis.

Research-Based Generative AI Tool
Python, Gemini API, Generative AI

Developed a research-oriented AI tool using the Gemini API to explore
practical applications of Generative AI for research and knowledge-based tasks.

ACHIEVEMENTS:
Solved 500+ Data Structures & Algorithms problems across multiple coding platforms.
LeetCode Contest Rating: 1614
Earned 10+ badges across multiple coding platforms.
""")


print("Name:", result.Name)
print("Sentiment:", result.sentiment)
print("Summary:", result.summary)
print("Skills:", result.Skills)
print("Projects:", result.Project)