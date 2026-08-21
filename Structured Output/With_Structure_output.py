from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.7, max_output_tokens=500)
class Review(TypedDict):
    summary: Annotated[str, "A concise summary of the input text."]
    sentiment: Annotated[Literal["positive","nagative","neutral"], "The sentiment of the input text"]
    Name:   Annotated[str,"Give me Name in the Text"]
    Education: Annotated[str, "The educational background of the person who wrote the input text."]
    Skills: Annotated[Optional [list[str]], "The technical skills of the person who wrote the input text."]
    Project: Annotated[str,"Give me Project list only name and techStack"]

structured_output = model.with_structured_output(Review)

result= structured_output.invoke("""
  This is Aditya Raj resume.B.Tech Computer Science Engineering student with strong foundations in Java, Python, Data Structures & Algorithms, Object-Oriented Programming, and SQL, with 500+ DSA problems solved across multiple coding platforms. Developed software applications using Flask, REST APIs, Machine Learning, and Generative AI, including a research-based AI tool built using the Gemini API. Currently exploring Generative AI fundamentals and interested in building practical AI-powered applications, scalable software systems, and solving real-world engineering problems. | EDUCATION: Galgotias University — B.Tech in Computer Science Engineering | Oct 2023–Oct 2027 | CGPA: 8.1/10 | St. Joseph's Sr. Sec. School — Class XII (CBSE) | May 2023 | 68% | D.A.V. Public School — Class X (CBSE) | April 2021 | 72% | TECHNICAL SKILLS: Languages: Java, Python, SQL | AI / Machine Learning: Machine Learning, NLP, Generative AI (Fundamentals), TF-IDF, Random Forest, Linear Regression | Backend: Flask, REST APIs | Databases: MySQL, SQLite | Core CS: Data Structures & Algorithms, OOP, Operating Systems, Computer Networks, System Design Basics | Developer Tools: Git, GitHub, VS Code | PROJECTS: AI-Powered Smart Expense Tracker | Python, Flask, SQLite, Machine Learning | Built an AI-powered expense management application using Flask, SQLite, and machine learning for automated expense categorization and spending analysis. | Trained and evaluated a TF-IDF + Random Forest model for automatic categorization, and implemented Z-score anomaly detection and Linear Regression forecasting to identify irregular transactions and generate spending insights. | Designed a Flask REST API with 8 endpoints integrating CRUD operations, ML inference, and a multi-page dashboard, with Joblib lazy loading for efficient model usage. | Research-Based Generative AI Tool | Python, Gemini API, Generative AI | Developed a research-oriented AI tool using the Gemini API to explore practical applications of Generative AI for research and knowledge-based tasks. | Integrated a large language model through an API-based workflow and experimented with prompts to generate AI-assisted responses for the selected research problem. | Built the project as part of an ongoing Generative AI learning journey, focusing on fundamental concepts, API integration, experimentation, and practical AI application development. | CERTIFICATIONS: AICTE–EduSkills Cloud Virtual Internship — AWS Academy | Oct–Dec 2024 | Java Programming Certification — GUVI | Database Programming with SQL — Oracle Academy | Dec 2024 | NPTEL Certified in Design Thinking — Elite + Silver | ACHIEVEMENTS: Solved 500+ Data Structures & Algorithms problems across multiple coding platforms. | LeetCode Contest Rating: 1614 | Earned 10+ badges across multiple coding platforms.

 """)

print(result)
print(result['Name'])

print(result['sentiment'])
