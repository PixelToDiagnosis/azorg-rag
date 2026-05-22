# Updated Advice: Project 1 - Radiology Knowledge Assistant with RAG

This document is a revised version of the earlier project advice for creating a RAG system. It keeps the original project idea - an internal-documents RAG chatbot for a radiology group - and adds the recent improvements discussed around CSV files, Google Sheets exports, custom MRI protocol loading, metadata, and safer chunking.

---

## 1. Goal

Build a small local RAG system that answers questions from radiology reference documents and always shows sources.

Use it first with fake, public, de-identified, or non-sensitive documents such as:

- MRI/CT protocols
- contrast guidelines
- MRI safety rules
- biopsy preparation instructions
- structured reporting templates
- incidentaloma follow-up policies
- administrative SOPs

Do not start with patient data, PACS/RIS integration, DICOM routing, or live clinical decision support. Identifiable health data requires governance, approval, hosting decisions, access controls, and privacy review.

---

## 2. Updated architecture

The original architecture was:

```text
Documents
  ↓
Document loader
  ↓
Text splitter / chunker
  ↓
Embeddings
  ↓
ChromaDB vector store
  ↓
Retriever
  ↓
Prompt with retrieved context
  ↓
LLM answer with citations/sources
```

The improved version should be:

```text
Raw documents
  ├── PDFs
  ├── Markdown / text
  └── CSV files / Google Sheets exports
        ↓
Document-specific loaders
        ↓
Normalize into LangChain Documents
        ↓
Add metadata
        ↓
Chunk only when appropriate
        ↓
Embed with OpenAI embeddings
        ↓
Store in ChromaDB
        ↓
Retrieve relevant documents
        ↓
Answer with sources
        ↓
Evaluate with test questions
```

The important improvement is this:

> Do not load every file type the same way.

PDFs, Markdown files, and CSV protocol tables need different loading strategies.

---

## 3. Recommended file formats

For your project:

| Source type | Recommended format | Loading strategy |
|---|---:|---|
| Protocol PDFs | PDF | `PyPDFLoader`, then chunk |
| Guidelines / notes | Markdown or TXT | Load as text, then chunk |
| Google Sheets with simple rows | CSV | One row = one document |
| MRI protocol tables | CSV | One protocol = one document |
| Word documents | DOCX or converted Markdown | Preferably convert to Markdown |
| Slides | PDF or extracted text | Use only if needed |

Do not export Google Sheets to PDF unless the layout itself is important. For RAG, CSV is much cleaner.

---

## 4. Key improvement: CSV files should not be treated like PDFs

For normal prose documents, chunking is useful.

For structured CSV files, chunking can damage the meaning.

Example problem:

```text
Protocol: Parkinson
Sequence: Ax T2
```

If this gets split badly, the model may retrieve `Ax T2` without knowing it belongs to the Parkinson protocol.

For your MRI protocol CSV, the best semantic unit is:

```text
one MRI protocol = one LangChain Document
```

not:

```text
one CSV row = one LangChain Document
```

and not:

```text
one page-like PDF chunk = one LangChain Document
```

---

## 5. Updated package structure

The original advice recommended a clean package structure with `loaders.py`, `chunking.py`, `vectorstore.py`, `rag.py`, and scripts such as `ingest.py` and `ask.py`.

I would now expand it slightly:

```text
radiology-ai-lab/
│
├── data/
│   ├── raw/
│   │   ├── pdf/
│   │   ├── csv/
│   │   └── markdown/
│   ├── processed/
│   └── chroma_db/
│
├── notebooks/
│   ├── 01_inspect_documents.ipynb
│   ├── 02_test_retrieval.ipynb
│   └── 03_evaluate_answers.ipynb
│
├── scripts/
│   ├── ingest.py
│   ├── ask.py
│   └── inspect_retrieval.py
│
├── src/
│   └── radai/
│       ├── __init__.py
│       ├── config.py
│       ├── loaders.py
│       ├── chunking.py
│       ├── vectorstore.py
│       ├── rag.py
│       ├── prompts.py
│       ├── safety.py
│       └── evaluation.py
│
├── tests/
│   ├── test_loaders.py
│   ├── test_chunking.py
│   └── test_rag.py
│
├── .env
├── requirements.txt
└── README.md
```

The mental model remains the same:

```text
scripts/ = things you run
src/radai/ = reusable helper functions
tests/ = checks that your code still works
data/ = your documents and vector database
notebooks/ = experimentation
```

---

## 6. Updated dependencies

Install the core packages:

```bash
pip install langchain langchain-openai langchain-community langchain-chroma chromadb pypdf python-dotenv pandas
```

Use OpenAI embeddings:

```python
OpenAIEmbeddings(model="text-embedding-3-small")
```

This remains a sensible default for your prototype.

For the chat model, use a low-cost fast model first. The earlier advice used `gpt-4o-mini`. Today, I would check your available API models and likely use a current mini model such as `gpt-4.1-mini` or another current low-cost model available in your account.

---

## 7. Updated `loaders.py`

This file should be only for loading documents.

### 7.1 PDF loader

```python
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(folder_path: str):
    documents = []

    pdf_files = Path(folder_path).glob("*.pdf")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_type"] = "pdf"
            doc.metadata["source_file"] = pdf_file.name

        documents.extend(docs)

    return documents
```

### 7.2 Generic CSV loader

Use this for simple 2-3 column Google Sheets where each row is a standalone fact.

```python
import csv
from pathlib import Path

from langchain_core.documents import Document


def load_simple_csv_documents(folder_path: str):
    documents = []

    csv_files = Path(folder_path).glob("*.csv")

    for csv_file in csv_files:
        with open(csv_file, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row_number, row in enumerate(reader, start=1):
                page_content = "\n".join(
                    f"{column}: {value}"
                    for column, value in row.items()
                    if value
                )

                document = Document(
                    page_content=page_content,
                    metadata={
                        "source_type": "csv",
                        "document_type": "simple_table",
                        "source_file": csv_file.name,
                        "row": row_number,
                    },
                )

                documents.append(document)

    return documents
```

This is good for tables like:

```csv
term,definition
DWI,Diffusion-weighted imaging
FLAIR,Fluid-attenuated inversion recovery
```

### 7.3 Custom MRI protocol CSV loader

For your MRI protocol document, use a custom loader.

The goal is to transform the CSV into this kind of document:

```text
Protocol: Parkinson

Indications:
- vermoeden Parkinson

Sequences:
- 2D FLAIR
- Ax T2
- Ax DWI
- 3D T1
- Ax T2*
- Evt. SWAN nigrosome
```

Suggested loader:

```python
import csv
from pathlib import Path

from langchain_core.documents import Document


def load_mri_protocol_csv(file_path: str):
    documents = []

    with open(file_path, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    version = rows[0][0].strip() if rows else ""
    authors = rows[1][0].strip() if len(rows) > 1 else ""

    current_protocol = None
    current_indications = []
    current_notes = []
    current_sequences = []

    def save_current_protocol():
        if current_protocol is None:
            return

        content = f"Protocol: {current_protocol}\n\n"

        if current_indications:
            content += "Indications:\n"
            for indication in current_indications:
                content += f"- {indication}\n"
            content += "\n"

        if current_notes:
            content += "Notes:\n"
            for note in current_notes:
                content += f"- {note}\n"
            content += "\n"

        if current_sequences:
            content += "Sequences:\n"
            for sequence in current_sequences:
                content += f"- {sequence}\n"

        document = Document(
            page_content=content.strip(),
            metadata={
                "source_type": "csv",
                "document_type": "mri_protocol",
                "source_file": Path(file_path).name,
                "version": version,
                "authors": authors,
                "protocol": current_protocol,
            },
        )

        documents.append(document)

    def is_protocol_title(text: str):
        lower = text.lower()

        non_title_starts = (
            "indicatie",
            "indicaties",
            "opmerking",
            "opmerkingen",
            "indien",
            "enkel",
            "-",
        )

        return text and not lower.startswith(non_title_starts)

    for row in rows[4:]:
        first_col = row[0].strip() if len(row) > 0 else ""
        second_col = row[1].strip() if len(row) > 1 else ""

        if not first_col and not second_col:
            continue

        if is_protocol_title(first_col):
            save_current_protocol()

            current_protocol = first_col
            current_indications = []
            current_notes = []
            current_sequences = []

            if second_col:
                current_sequences.append(second_col)

            continue

        if first_col.lower().startswith(("indicatie", "indicaties")):
            indication = first_col.split(":", 1)[-1].strip()
            if indication:
                current_indications.append(indication)

        elif first_col:
            current_notes.append(first_col)

        if second_col:
            current_sequences.append(second_col)

    save_current_protocol()

    return documents
```

### 7.4 Combined loader

```python
def load_all_documents():
    documents = []

    documents.extend(load_pdf_documents("./data/raw/pdf"))

    documents.extend(
        load_mri_protocol_csv(
            "./data/raw/csv/AZORG MRI Neuro Protocols.csv"
        )
    )

    return documents
```

Later, you can make this more flexible, but explicit loading is easier while learning.

---

## 8. Updated `chunking.py`

The original document suggested a standard text splitter with `chunk_size=1000` and `chunk_overlap=150`.

Keep that for PDFs and long text.

But skip chunking for MRI protocol CSV documents.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=150,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def split_documents_by_type(documents):
    documents_to_chunk = []
    documents_to_keep = []

    for document in documents:
        source_type = document.metadata.get("source_type")
        document_type = document.metadata.get("document_type")

        if source_type == "csv" and document_type == "mri_protocol":
            documents_to_keep.append(document)
        elif source_type == "csv" and document_type == "simple_table":
            documents_to_keep.append(document)
        else:
            documents_to_chunk.append(document)

    chunks = split_documents(documents_to_chunk)

    return chunks + documents_to_keep
```

This is one of the most important improvements.

For your MRI protocols:

```text
one protocol = one document
```

For PDFs:

```text
one long page/document = multiple chunks
```

---

## 9. Updated `vectorstore.py`

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def get_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def build_vectorstore(
    documents,
    persist_directory="./data/chroma_db",
    collection_name="radiology_knowledge",
):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    return vectorstore


def load_vectorstore(
    persist_directory="./data/chroma_db",
    collection_name="radiology_knowledge",
):
    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    return vectorstore
```

---

## 10. Updated `rag.py`

The original version joined retrieved documents into a prompt and asked the model to answer only from the context.

Improve it by formatting sources clearly.

```python
from langchain_openai import ChatOpenAI


def format_docs_with_sources(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown source")
        protocol = doc.metadata.get("protocol")

        source_label = source
        if protocol:
            source_label += f" | protocol: {protocol}"

        formatted.append(
            f"[Source {i}: {source_label}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def answer_question(query, retriever):
    docs = retriever.invoke(query)

    context = format_docs_with_sources(docs)

    prompt = f"""
You are a radiology knowledge assistant.

Answer the question using ONLY the context below.

Rules:
- If the answer is not in the context, say:
  "I could not find this in the indexed documents."
- Always mention which source(s) you used.
- Do not invent protocols.
- Do not provide patient-specific medical advice.
- Keep the answer concise and practical.

Context:
{context}

Question:
{query}
"""

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    response = llm.invoke(prompt)

    return response.content
```

---

## 11. Updated `scripts/ingest.py`

```python
from radai.loaders import load_all_documents
from radai.chunking import split_documents_by_type
from radai.vectorstore import build_vectorstore


docs = load_all_documents()

chunks = split_documents_by_type(docs)

vectorstore = build_vectorstore(chunks)

print(f"Loaded {len(docs)} documents.")
print(f"Indexed {len(chunks)} chunks/documents.")
print("Documents indexed successfully.")
```

---

## 12. Updated `scripts/ask.py`

```python
from radai.vectorstore import load_vectorstore
from radai.rag import answer_question


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

question = input("Ask a question: ")

answer = answer_question(question, retriever)

print("\nANSWER:\n")
print(answer)
```

---

## 13. Recommended retrieval settings

Start simple:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
```

Then test:

```python
k = 3
k = 5
k = 8
```

For your MRI protocol CSV, I would start with:

```python
k = 4
```

because each retrieved document may already be a complete protocol.

---

## 14. Example questions to test

For the MRI protocol document:

```text
Welke sequenties zijn nodig voor Parkinson?
```

```text
Wat is het MRI protocol voor dementie?
```

```text
Welke indicaties horen bij het MS protocol?
```

```text
Welk protocol gebruik ik bij vermoeden hypofyseletsel?
```

```text
Welke sequenties zitten in het screening protocol?
```

Also test “not found” behavior:

```text
Wat is het protocol voor CT thorax?
```

The correct behavior should be something like:

```text
I could not find this in the indexed documents.
```

---

## 15. Add an inspection script

This is very useful while learning.

Create:

```text
scripts/inspect_retrieval.py
```

```python
from radai.vectorstore import load_vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

question = input("Test retrieval for question: ")

docs = retriever.invoke(question)

for i, doc in enumerate(docs, start=1):
    print("\n" + "=" * 80)
    print(f"RESULT {i}")
    print("Metadata:")
    print(doc.metadata)
    print("\nContent:")
    print(doc.page_content[:1500])
```

Run:

```bash
python scripts/inspect_retrieval.py
```

This helps you see whether the retriever finds the right protocol before you blame the LLM.

---

## 16. Add metadata deliberately

For every document, include metadata such as:

```python
{
    "source_type": "csv",
    "document_type": "mri_protocol",
    "source_file": "AZORG MRI Neuro Protocols.csv",
    "protocol": "Parkinson",
    "version": "Versie 30/7/2025"
}
```

Metadata helps with:

- source citations
- debugging
- filtering later
- knowing whether a result came from a PDF or CSV
- showing protocol names in answers

---

## 17. Updated learning path

The original advice proposed a sensible path: package skeleton, ingestion helpers, Chroma, RAG chatbot, citations, chat history, evaluation, structured extraction, tests, and Streamlit.

I would now update the order like this:

1. Create Python package skeleton.
2. Load PDFs with `PyPDFLoader`.
3. Load CSVs with custom loaders.
4. For MRI protocols, create one protocol = one document.
5. Add metadata.
6. Chunk only PDFs/long text, not protocol CSVs.
7. Build Chroma vector store.
8. Add retrieval inspection script.
9. Add answer generation with source display.
10. Create 20-30 test questions.
11. Add simple evaluation.
12. Add Streamlit interface.
13. Only later add chat history.
14. Only later add structured extraction as Project 2.

I would not add chat history too early. First make retrieval reliable.

---

## 18. Minimal milestone plan

### Milestone 1 - Local ingestion

Success means:

```text
python scripts/ingest.py
```

loads your PDFs and CSVs into Chroma without errors.

### Milestone 2 - Retrieval inspection

Success means:

```text
python scripts/inspect_retrieval.py
```

finds the correct MRI protocol for common questions.

### Milestone 3 - RAG answers

Success means:

```text
python scripts/ask.py
```

answers with:

- relevant protocol
- listed sequences
- source file
- no hallucinated details

### Milestone 4 - Evaluation set

Create a small CSV:

```csv
question,expected_source,expected_protocol,expected_keywords
Welke sequenties voor Parkinson?,AZORG MRI Neuro Protocols.csv,Parkinson,"DWI;T2;3D T1"
Wat is het screening protocol?,AZORG MRI Neuro Protocols.csv,Screening,"FLAIR;DWI;3D T1"
```

Then later write a script that checks whether the retrieved documents contain the expected protocol.

---

## 19. Updated project concept

Your Project 1 should now be:

```text
Radiology Knowledge Assistant

Input:
- PDF protocols
- Markdown notes
- CSV exports from Google Sheets
- custom MRI protocol tables

Core behavior:
- load documents intelligently
- preserve structure
- embed with metadata
- retrieve relevant context
- answer only from retrieved sources
- say "not found" when needed
```

Updated modules:

```text
radai.loaders      Load PDFs, Markdown, CSVs, MRI protocol tables
radai.chunking     Split only documents that need splitting
radai.vectorstore  Build/load ChromaDB
radai.rag          Answer questions from retrieved context
radai.prompts      Store prompt templates
radai.evaluation   Test retrieval and answer quality
radai.safety       Guardrails and "not found" behavior
```

This preserves the original advice but makes it more robust and more suited to your actual documents.

The biggest practical improvement is:

> For structured medical protocol tables, design the loader around the clinical meaning, not the file format.

For your MRI protocol CSV, that means:

```text
one protocol = one document
```

with clear metadata and no extra chunking.
