import os
from pathlib import Path
parent_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
faiss_vectorstore_dir = Path(os.path.join(parent_dir, 'faiss_vectorstore_object'))


import streamlit as st
import llm_model
import chat_prompt_template
from loguru import logger
from typing import Optional, Any

from langchain_core.messages import AIMessage, HumanMessage
import joblib

from dotenv import load_dotenv
load_dotenv()

from document_loader import web_document_loader


st.set_page_config(page_title="Q&A with any website")

st.header("LangChain Application")

input=st.text_input("Enter the URL of the website you want to index: ", key="input")

question = st.text_input("Ask a question about the website: ")

# display the website in focus to the user
st.subheader(f"{input}")

submit=st.button("Enter")


def langsmith_app(prompt:str, site_link:str,):
    " a question and answer function when given a website and a query prompt"
    if site_link and prompt:

        # call the web loader module
        logger.info('loading the website document')
        docs = web_document_loader(url = site_link)

        # call the llm model module
        logger.info('loading the LLM model')
        llm = llm_model.load_model(model_name = "llama3")

        # load the embedding layer module
        logger.info('generating the embeddings for the website contents...')
        vector  = llm_model.generate_embeddings(model_name="llama3", loaded_document=docs)

        logger.info('embeddings generated and saved!')
        joblib.dump(vector, f"{faiss_vectorstore_dir}/vectorstore.pkl")

        # call the chat prompt template module
        logger.info('creating the retrieval chain')
        retrieval_chain = chat_prompt_template.retrieval_chain(
            llm_model=llm,
            vector_object=vector
        )
        logger.info('Response is being retrieved!')
        response =  retrieval_chain.invoke(
            {
                "input": prompt
            }
            )
        return response['answer']



def langsmith_app_with_chat_history(prompt:str, chat_history:Optional[list | None], contexts:str, vector_object):
    """ a method to provide continuous response based on your conversation history"""
    if prompt:
        # call the llm model module
        logger.info('loading the LLM model')
        llm = llm_model.load_model(model_name = "llama3")

        # call the chat prompt template module
        logger.info('creating the retrieval chain')
        retrieval_chain_with_history = chat_prompt_template.retrieval_chain_with_history(
            llm_model=llm,
            vector_object=vector_object
        )
        logger.info('Response is being retrieved!')
        response =  retrieval_chain_with_history.invoke(
            {
                "chat_history": chat_history,
                "input": prompt,
                "context": contexts
            }
            )
        return response['answer']




if __name__ == "__main__":

    # if user enters a valid website and a prompt
    if input:
        if submit:
            response = langsmith_app(prompt=question, site_link=input,)
            st.subheader("The Response is")
            st.write(response)
    
        
            

