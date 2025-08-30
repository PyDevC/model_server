from memory.ragpipe import run_rag_pipeline
import sys
from hf_model.model_cards.deepseek_coder import (
    DeepSeek_Coder_1_3B_Instruct
)

def main():
    """
    An interactive command-line interface to run the full RAG pipeline.
    This function will prompt the user for a source and a question,
    then use the custom offline model to generate an answer.
    """
    print("--- Model Server CLI ---")
    print("Type 'quit' at any prompt to exit the application.")
    print("-" * 50)

    while True:
        source_query = input("\nEnter the document source (URL, file/directory path, or a web search term):\n> ")
        if source_query.lower() == 'quit':
            break

        question = input("\nNow, enter the question you want to ask about this source:\n> ")
        if question.lower() == 'quit':
            break

        print("\nProcessing your request... This may take a moment.")
        
        answer = run_rag_pipeline(
            model=DeepSeek_Coder_1_3B_Instruct(),
            source_query=source_query,
            message=question,
        )

        print("\n" + "="*20 + " Answer " + "="*20)
        print(answer)
        print("=" * 48 + "\n")

    print("\nExiting the RAG Pipeline CLI. Goodbye!")

if __name__ == "__main__":
    main()
