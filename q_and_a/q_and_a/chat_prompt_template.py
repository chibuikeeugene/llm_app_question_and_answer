# laoding a template to help refine our model response
from langchain_core.prompts import ChatPromptTemplate

# method to chain our prompt template and the llm model
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# importing functions to help create and handle chat history
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder


from langchain.chains import create_retrieval_chain

def prompt_template(llm_model, vector_object):
    """ customize the template prompt"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a renowned customer support engineer. Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    # create a document chain that holds user input prompt and the model
    document_chain = create_stuff_documents_chain(prompt=prompt, llm=llm_model)
    

    # creating a retriever object from the vectorstore created in the llmmodel module
    retriever = vector_object.as_retriever()

    # creating chat history retriever object
    retriever_chain_with_history = create_history_aware_retriever(llm =llm_model, prompt=prompt, retriever=retriever)

    # creating our retiever chain
    retrieval_chain_update = create_retrieval_chain(retriever_chain_with_history, document_chain)

    return retrieval_chain_update