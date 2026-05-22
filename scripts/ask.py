from radai.vectorstore import load_vectorstore
from radai.rag import answer_question


def main():
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()

    print("Creating retriever...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("Ready for questions.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower().strip() in ["exit", "quit", "q"]:
            print("Goodbye.")
            break

        if not question.strip():
            continue

        print("\nGenerating answer...\n")

        answer = answer_question(question, retriever)

        print("ANSWER:\n")
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
