from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)


def split_documents_by_type(documents):
    documents_to_chunk = []
    documents_to_keep = []

    for document in documents:
        source_type = document.metadata.get("source_type")
        document_type = document.metadata.get("document_type")

        if source_type == "csv" and document_type == "mri_ct_protocol":
            documents_to_keep.append(document)
        if source_type == "csv" and document_type == "simple_table":
            documents_to_keep.append(document)
        else:
            documents_to_chunk.append(document)

    chunks = split_documents(documents_to_chunk)

    return chunks + documents_to_keep
