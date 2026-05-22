from radai.vectorstore import load_vectorstore


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

question = input("Test retrieval for question: ")

docs = retriever.invoke(question)

for i, doc in enumerate(docs, start=1):
    print("\n" + "=" * 80)
    print(f"RESULT {i}")
    print("Metadata:")
    print(doc.metadata)
    print("\nContent:")
    print(doc.page_content[:1500])
