# laoding a template to help refine our model response
from langchain_core.prompts import ChatPromptTemplate


def prompt_template():
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
        ]
    )
    return template