from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import csv


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


def load_markdown_documents(folder_path: str):
    documents = []

    md_files = Path(folder_path).glob("*md")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as file:
            text = file.read()

        document = Document(
            page_content=text,
            metadata={"source_type": "markdown", "source_file": md_file.name},
        )
        documents.append(document)

    return documents


def load_simple_csv_documents(folder_path: str):
    documents = []

    csv_files = Path(folder_path).glob("*.csv")

    for csv_file in csv_files:
        with open(csv_file, newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            for row, number in enumerate(reader, start=1):
                page_content = "\n".join(
                    f"{column}: {value}" for column, value in row.items() if value
                )

                document = Document(
                    page_content=page_content,
                    metadata={"source_type": "csv", "source_file": csv_file.name},
                )

                documents.append(document)

    return documents


def load_mri_ct_protocol_csv(file_path: str):
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
                "document_type": "mri_ct_protocol",
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


def load_all_documents():
    documents = []

    documents.extend(load_pdf_documents("./data/raw/pdf"))
    documents.extend(load_markdown_documents("./data/raw/markdown"))
    documents.extend(
        load_mri_ct_protocol_csv("./data/raw/csv/azorg_mri_cardio_protocols.csv")
    )
    documents.extend(
        load_mri_ct_protocol_csv("./data/raw/csv/azorg_mri_neuro_protocols.csv")
    )

    return documents
