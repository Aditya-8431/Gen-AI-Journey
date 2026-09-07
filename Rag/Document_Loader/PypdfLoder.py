from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai  import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# Update this path if the PDF is stored elsewhere
loader = PyPDFLoader(r"C:\Users\neela\OneDrive\Desktop\LangChain\Rag\Document_Loader\Accenture_technical_important.pdf")

docs=loader.load()

print("length of loader" + str(len(docs)))

