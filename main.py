# file_extract.py
import os
import argparse
import getpass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

from file_extract import extract_content


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter your API key: ")

    parser = argparse.ArgumentParser(
        description="Process a text file and generate relevant content."
    )
    parser.add_argument("file_path", help="Path to the input text file")

    args = parser.parse_args()

    # FetchDB
    retriever = extract_content(args.file_path)

    # LLMs
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
    input_prompt = input("Enter your prompt: ")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
        ]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(input_prompt)


if __name__ == "__main__":
    main()
