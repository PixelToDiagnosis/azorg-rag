from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

embedding_function = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key,
)


def build_vectorstore(
    documents,
    persist_directory="./data/chroma_db",
    collection_name="radiology_knowledge",
):
    vectorstore = Chroma.from_documents(
        documents=documents,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding=embedding_function,
    )
    return vectorstore


def load_vectorstore(
    persist_directory="./data/chroma_db",
    collection_name="radiology_knowledge",
):

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    return vectorstore
