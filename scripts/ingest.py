from radai.loaders import load_all_documents
from radai.chunking import split_documents_by_type
from radai.vectorstore import build_vectorstore

docs = load_all_documents()
chunks = split_documents_by_type(docs)
vectorstore = build_vectorstore(chunks)

print(f"Loaded {len(docs)} documents.")
print(f"Indexed {len(chunks)} chunks/documents.")
print("Documents indexed succesfully.")
