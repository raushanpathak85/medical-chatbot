## Import the library
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings



## Extract the text from the PDF files.
def load_pdf_files(Data):
    loader= DirectoryLoader(Data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents


## Filter the minimal docs operation
def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    """
    Given a list of document object, return a new list of document object containing only 'source'
    in metadeta and the original page content.
    """
    minimal_docs:List[Document]= []

    for doc in docs:
        src=doc.metadata.get('source')
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    
    return minimal_docs

## Split the documents into smaller chunks
def text_split(minimal_docs):
    text_split=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20,length_function=len)
    text_chunk=text_split.split_documents(minimal_docs)

    return text_chunk

## Create the Embedding models
def download_embedding():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embedding=HuggingFaceEmbeddings(model_name=model_name)
    return embedding

