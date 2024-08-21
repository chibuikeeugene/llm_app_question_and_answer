# laoding a template to help refine our model response
from langchain_core.prompts import ChatPromptTemplate

# method to chain our prompt template and the llm model
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain


from langchain.chains import create_retrieval_chain

def prompt_template(llm_model, vector_object):
    """ customize the template prompt"""
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
                                            
            Question: {input}"""
    )


    document_chain = create_stuff_documents_chain(prompt=prompt, llm=llm_model)
    

    # creating a retriever object from the vectorstore created in the llmmodel module
    retriever = vector_object.as_retriever()
    # creating our retiever chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain