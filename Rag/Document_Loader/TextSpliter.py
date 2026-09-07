from langchain_community.document_loaders import TextLoader
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

prompt= PromptTemplate(
    template='Write the summary of following poem -\n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

loader= TextLoader('Cricket.txt' ,encoding='utf-8')
#giving memory

docs =loader.load()

print(type(docs))
print(docs[0].metadata)

chain= prompt|model|parser
print(chain.invoke(({'text':docs[0].page_content})))