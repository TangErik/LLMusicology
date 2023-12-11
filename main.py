# file_extract.py
import os
import argparse
import getpass
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma

from file_extract import extract_content

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter your API key: ")

    # FetchDB
    if os.path.isdir("/vector_dir") == False:
        print("No folder found, extracting content from file.")
        file_path = input("Path to the input text file: ")
        extract_content(file_path)
    
    # Retriever
    vectorstore = Chroma(persist_directory="/vector_dir", embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

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

    response = rag_chain.invoke(input_prompt)
    print("Response:", response)
    return response

if __name__ == "__main__":
    main()
