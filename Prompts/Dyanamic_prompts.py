import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --------------------------------
# Create Gemini Model
# --------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# --------------------------------
# Streamlit UI
# --------------------------------

st.header("Research Tool")
st.write("Model: Gemini 3.6 Flash")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

# --------------------------------
# Dynamic Prompt
# --------------------------------

template = """
Please Summarize the research paper titled {paper_input} with the following specifications:
Explanation Style: {style_input}

Explanation Length: {length_input}

1. Mathematical Details:
   -Include relevant mathematical equations if present in the paper.
    -Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
 -Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.

"""


prompt = PromptTemplate(
    template=template,
    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ]
)

# --------------------------------
# Submit
# --------------------------------

if st.button("Submit"):

    # Create dynamic prompt
    final_prompt = prompt.format(
        paper_input=paper_input,
        style_input=style_input,
        length_input=length_input
    )

    # Ask Gemini
    response = model.invoke(final_prompt)

    # Display response
    st.subheader("Gemini's Explanation")

    if isinstance(response.content, str):
        st.markdown(response.content)
    else:
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                st.markdown(block.get("text", ""))