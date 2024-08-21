from langchain_community.document_loaders import WebBaseLoader

# A function to load the web documents
def web_document_loader(url):
    """ web document loader."""
    loader = WebBaseLoader(url)
    # call the load function of the webbasedloader
    docs = loader.load()
    
    return docs