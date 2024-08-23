import fire
import streamlit as st
import llm_model
import chat_prompt_template
from loguru import logger

from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

from document_loader import web_document_loader

# define global variables
chat_history:list = []

def langsmith_app(prompt:str, site_link):

    # retrieve url of any website
    # site_link =  input("Enter the URL of the website you want to index: ")

    # call the web loader module
    logger.info('loading the website document')
    docs = web_document_loader(url = site_link)

    # call the llm model module
    logger.info('loading the LLM model')
    llm = llm_model.load_model(model_name = "llama3")

    # load the embedding layer module
    logger.info('generating the embeddings for the website contents...')
    vector  = llm_model.generate_embeddings(model_name="llama3", loaded_document=docs)

    # call the chat prompt template module
    logger.info('creating the retrieval chain')
    retrieval_chain_with_history = chat_prompt_template.prompt_template(
        llm_model=llm,
        vector_object=vector
    )
    logger.info('Response is being retrieved!')
    response =  retrieval_chain_with_history.invoke(
        {
            "chat_history": chat_history,
            "input": prompt
        }
        )
    return response['answer']


if __name__ == "__main__":
    st.set_page_config(page_title="Q&A with any website")

    st.header("LangChain Application")

    input=st.text_input("Enter the URL of the website you want to index: ", key="input")
    
    question = st.text_input("Ask a question about the website: ")

    # display the website in focus to the user
    st.subheader(f"{input}")

    submit=st.button("Enter")

    if submit:
        response = langsmith_app(prompt=question, site_link=input)
        st.subheader("The Response is")
        st.write(response)
        chat_history = [
            HumanMessage(
                content= f"{question}"
            ), 
            AIMessage(
                content= f"{response}"
            )
        ]
            
            

    

