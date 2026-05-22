from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)


def format_docs_with_sources(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown source")
        protocol = doc.metadata.get("protocol")

        source_label = source
        if protocol:
            source_label += f"protocol | {protocol}"

        formatted.append(f" source {i}: {source_label}|\n{doc.page_content}")
    return "\n\n".join(formatted)


def format_unique_sources(docs):
    unique_sources = set()

    for doc in docs:
        source = doc.metadata.get("source_file", "unknown source")
        protocol = doc.metadata.get("protocol")

        if protocol:
            source_label = f"{source} | protocol: {protocol}"
        else:
            source_label = source

        unique_sources.add(source_label)

    sources_text = "\n".join(f"- {source}" for source in sorted(unique_sources))

    return sources_text


def answer_question(query, retriever):
    docs = retriever.invoke(query)

    context = format_docs_with_sources(docs)
    sources_text = format_unique_sources(docs)

    prompt = f"""
You are a radiology knowledge assistant.

Answer the question using ONLY the context below.

Rules:
- If the answer is not in the context, say:
  "I could not find this in the indexed documents."
- Do not invent protocols.
- Do not provide patient-specific medical advice.
- Keep the answer concise and practical.
- Do not write "source 1", "source 2", or "source 3".
- At the end, list the sources exactly as provided below.

Context:
{context}

Available sources:
{sources_text}

Question:
{query}
"""

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    response = llm.invoke(prompt)

    return response.content
