from typing import Any
from .browse import get_documents_for_query, create_vector_store_retriever

# Core LangChain imports for building the RAG chain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.language_models.llms import LLM

class CustomHFModel(LLM):
    model: Any
    """
    A custom LangChain LLM wrapper for inferencing hf_models
    """
    @property
    def _llm_type(self) -> str:
        return "huggingface model"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        return self.model.generate(prompt)


def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, llm):
    """
    Creates and returns a complete RAG chain.
    """
    # This prompt template is designed to instruct the LLM to answer the question
    # based *only* on the provided context.
    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise.

    Context: {context} 

    Question: {question} 

    Answer:
    """
    prompt = PromptTemplate.from_template(template)

    # This chain uses LangChain Expression Language (LCEL) to define the data flow.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def run_rag_pipeline(model, source_query: str, message: str,):
    """
    Executes the full RAG pipeline from document loading to answer generation.

    Returns:
        The generated answer string, or an error message.
    """
    documents = get_documents_for_query(source_query)
    if not documents:
        return "Failed to load documents. Please check the source query."

    retriever = create_vector_store_retriever(documents)
    if not retriever:
        return "Failed to create the vector store retriever."
    
    llm = CustomHFModel(model=model) 
    rag_chain = create_rag_chain(retriever, llm)
    
    try:
        answer = rag_chain.invoke(message)
        return answer
    except Exception as e:
        return f"An error occurred while running the RAG chain: {e}"
