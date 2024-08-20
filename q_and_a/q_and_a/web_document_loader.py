from langchain_community.document_loaders import WebBaseLoader


def web_document_loader(url):
    # load the web based document
    loader = WebBaseLoader("{url}")
    docs = loader.load()