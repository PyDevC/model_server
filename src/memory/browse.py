import os
import re
from typing import List, Union
from urllib.parse import quote_plus

from .docmanagement import (
    textloader,
    pdfloader,
    csvloader,
    webloader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

URL_REGEX = re.compile(
    r"^(?:http|ftp)s?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$", re.IGNORECASE)

def is_url(text: str) -> bool:
    """Checks if a given string is a URL."""
    return re.match(URL_REGEX, text) is not None

def get_documents_for_query(query: str, directory_glob_pattern: str = "**/*") -> List[dict]:
    """
    Automatically determines the source from a query and loads documents.

    This function inspects the query to see if it's a URL, a local file path,
    or a directory path. If it's none of those, it treats the query as a
    search term and attempts to browse the internet for relevant information
    by constructing a search engine query URL.

    Args:
        query: The input string, which can be a URL, file path, directory path, or search term.
        directory_glob_pattern: The glob pattern to use when the query is a directory.
                                Defaults to "**/*" to load all files in all subdirectories.
                                Supported file types depend on the 'unstructured' library.

    Returns:
        A list of loaded documents. Returns an empty list if loading fails or the
        source type is not supported.
    """
    documents = []
    
    if is_url(query):
        documents = webloader(query)
    elif os.path.isfile(query):
        _, extension = os.path.splitext(query)
        extension = extension.lower()
        
        if extension == '.txt':
            documents = textloader(query)
        elif extension == '.pdf':
            documents = pdfloader(query)
        elif extension == '.csv':
            documents = csvloader(query)
    else:
        print(f"Detected search term. Browsing the web for: '{query}'")
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        print(f"Using search URL: {search_url}")
        documents = webloader(search_url)

    if not documents:
        print("Could not retrieve any documents for the query.")
        
    return documents

def create_vector_store_retriever(documents: List[dict], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Creates a vector store retriever from a list of documents.

    This function takes the loaded documents, splits them into smaller chunks,
    generates embeddings for these chunks using a sentence-transformer model,
    and then stores them in a FAISS vector store.

    Args:
        documents: A list of documents loaded by one of the loader functions.
        chunk_size: The maximum size of each text chunk (in characters).
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A retriever object that can be used to fetch relevant documents based
        on a query. Returns None if the process fails.
    """
    if not documents:
        return None

    print(f"Processing {len(documents)} documents for vector storage.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
    except Exception as e:
        print(e)

        
    if not texts:
        print("Document splitting resulted in no text chunks.")
        return None
        
    print(f"Created {len(texts)} text chunks.")

    print("Initializing embedding model (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Creating FAISS vector store from document chunks...")
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("Vector store created successfully.")
    except Exception as e:
        print(f"An error occurred while creating the vector store: {e}")
        return None

    return vectorstore.as_retriever()
