# from langchain_community.document_loaders import WebBaseLoader

# url="https://en.wikipedia.org/wiki/Artificial_intelligence"
# loader=WebBaseLoader(url)
# docs=loader.load()
# print(docs[0].page_content);

# This above block of code is used to load the content of a web page using the WebBaseLoader from the langchain_community.document_loaders module.
#  It specifies a URL (in this case, a product page on Flipkart) and creates a loader instance for that URL. 
# The loader then loads the content of the web page into a list of documents (docs), and the first document's page content is printed to the console.

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt= PromptTemplate(
    template='What is AI  write answer in 2 lines  -\n {text}',
    input_variables=['text']
)

parser=StrOutputParser()
url="https://en.wikipedia.org/wiki/Artificial_intelligence"
loader=WebBaseLoader(url)
docs=loader.load()

chain =prompt|model|parser
result=chain.invoke({'text':docs[0].page_content})
print(result)

# This above block of code is used to ask a question about AI based on the content of a web page using the WebBaseLoader from the langchain_community.document_loaders module.




