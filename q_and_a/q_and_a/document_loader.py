from langchain_community.document_loaders import WebBaseLoader
# import bs4

# A function to load the web documents
def web_document_loader(url):
    """ web document loader."""
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        url
        # bs_kwargs= {"parse_only": bs4_strainer}
        )
    # call the load function of the webbasedloader
    docs = loader.load()

    return docs