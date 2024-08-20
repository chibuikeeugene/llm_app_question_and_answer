# import packages
from langchain_community.llms import Ollama

from langchain_community.embeddings import OllamaEmbeddings

# We will use a simple local vectorstore, FAISS, for now
# For FAISS, install the library faiss-cpu
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_model(model_name:str):
    """ load the specific llm model"""

    # initialize our llm model
    llm = Ollama(model= model_name)

    return llm


def generate_embeddings(model_name:str,):
    # Indexing our docs into a vectorstore
    # first create our embedding object
    embeddings  = OllamaEmbeddings(model=model_name)

    # Building our index
    # instantiate our text splitter
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
