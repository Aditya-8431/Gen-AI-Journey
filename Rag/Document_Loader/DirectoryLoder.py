from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loder =DirectoryLoader(
    path='books' ,# path of folder
    glob ='*.pdf',
    loader_cls=PyPDFLoader

)

docs=loder.load()
print(docs[0].page_content)
