# import packages
from loguru import logger

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

    logger.info("LLM model loaded, returning llm object ...")
    return llm


def generate_embeddings(model_name:str, loaded_document):
    """ generate content embeddings and save it in a FAISS vectorstore (in-memeory)"""
    # Indexing our docs into a vectorstore
    # first create our embedding object
    docs =  loaded_document

    embeddings  = OllamaEmbeddings(model=model_name)

    # Building our index
    # instantiate our text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    logger.info('performing recursive document splitting')
    documents = text_splitter.split_documents(docs)

    logger.info('performing content embedding from the document')
    vector = FAISS.from_documents(documents, embeddings)

    # return our vectorstore with the embedded contents
    logger.info("Vectorstore created, returning vector object ...")
    return vector
