from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    max_output_tokens=500
)

resume_schema = {
    "title": "ResumeExtraction",
    "description": "Extract structured information from a candidate's resume.",
    "type": "object",
    "properties": {

        "name": {
            "type": "string",
            "description": "Full name of the candidate."
        },

        "summary": {
            "type": "string",
            "description": "A concise professional summary of the candidate."
        },

        "education": {
            "type": "array",
            "description": "Educational qualifications mentioned in the resume.",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {
                        "type": "string",
                        "description": "Name of the educational institution."
                    },
                    "degree": {
                        "type": "string",
                        "description": "Degree or qualification."
                    },
                    "field": {
                        "type": "string",
                        "description": "Field of study."
                    },
                    "duration": {
                        "type": "string",
                        "description": "Study duration or graduation period."
                    },
                    "score": {
                        "type": "string",
                        "description": "CGPA, percentage, or other academic score if available."
                    }
                },
                "required": [
                    "institution",
                    "degree",
                    "field",
                    "duration",
                    "score"
                ]
            }
        },

        "technical_skills": {
            "type": "object",
            "description": "Technical skills categorized according to the resume.",
            "properties": {
                "languages": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "ai_ml": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "backend": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "databases": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "core_cs": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "developer_tools": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "languages",
                "ai_ml",
                "backend",
                "databases",
                "core_cs",
                "developer_tools"
            ]
        },

        "projects": {
            "type": "array",
            "description": "Projects mentioned in the resume.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name."
                    },
                    "technologies": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Technologies and tools used in the project."
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the project."
                    }
                },
                "required": [
                    "name",
                    "technologies",
                    "description"
                ]
            }
        },

        "certifications": {
            "type": "array",
            "description": "Certifications mentioned in the resume.",
            "items": {
                "type": "string"
            }
        },

        "achievements": {
            "type": "array",
            "description": "Achievements, competitive programming records, awards, or badges.",
            "items": {
                "type": "string"
            }
        }
    },

    "required": [
        "name",
        "summary",
        "education",
        "technical_skills",
        "projects",
        "certifications",
        "achievements"
    ]
}


structured_output = model.with_structured_output(resume_schema)


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

print(result["name"])
print(result)