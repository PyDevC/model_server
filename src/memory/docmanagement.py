from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    WebBaseLoader
)

def load_doc(loader, docpath):
    try:
        loading = loader(docpath)
        doc = loading.load()
        return doc
    except Exception as e:
        print(e, "Loading empty doc")
        return []

def textloader(docpath):
    return load_doc(TextLoader, docpath)

def pdfloader(docpath):
    return load_doc(PyPDFLoader, docpath)

def csvloader(docpath):
    return load_doc(CSVLoader, docpath)

def webloader(url):
    return load_doc(WebBaseLoader, url)
