# file_extract.py
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


# Load the document, split it into chunks, embed each chunk and load it into the vector store.
def extract_content(file_path):
    raw_documents = TextLoader(file_path).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    # print("here's the documents", documents)
    vectorstore = Chroma.from_documents(
        documents, OpenAIEmbeddings(), persist_directory="output_dir"
    )
    # print("here's the vectorstore", vectorstore)
    retriever = vectorstore.as_retriever()
    return retriever
