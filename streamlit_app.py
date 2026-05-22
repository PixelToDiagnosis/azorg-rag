import streamlit as st

from radai.vectorstore import load_vectorstore
from radai.rag import answer_question


st.set_page_config(
    page_title="AZORG RAD",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def get_retriever():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever


st.title("Radiology Knowledge Assistant")
st.caption("Ask questions about your indexed radiology documents.")

retriever = get_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            answer = answer_question(question, retriever)

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
