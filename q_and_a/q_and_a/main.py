import fire
import streamlit as st
import llm_model
import chat_prompt_template
from loguru import logger

from document_loader import web_document_loader


def langsmith_app(url):
    
    # call the web loader module
    logger.info('loading the website document')
    docs = web_document_loader(url = url)

    # call the llm model module
    logger.info('loading the LLM model')
    llm = llm_model.load_model(model_name = "llama3")

    # load the embedding layer module
    logger.info('generating the embeddings')
    vector  = llm_model.generate_embeddings(model_name="llama3", loaded_document=docs)

    # call the chat prompt template module
    logger.info('creating the retrieval chain')
    retrieval_chain = chat_prompt_template.prompt_template(
        llm_model=llm,
        vector_object=vector
    )

    response =  retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    return response['answer']

    # # create a chat interface
    # st.title("Question and Answer App")
    # question = st.text_input("Enter your question:")
    # if st.button("Ask"):
    #     response = llm.invoke(prompt.format(question=question, context=docs))
    #     st.write(response)


if __name__ == "__main__":
    site_link =  input("Enter the URL of the website you want to index: ")
    print(site_link)
    result = langsmith_app(site_link)
    print(result)
