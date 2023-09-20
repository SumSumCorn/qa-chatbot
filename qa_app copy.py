import os
import shutil
from dotenv import load_dotenv
import streamlit as st
import fitz

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


def ensure_temp_directory():
    if not os.path.exists("./temp"):
        os.makedirs("./temp")


def save_uploaded_file(uploaded_file):
    with open(f"./temp/{uploaded_file.name}", "wb") as buffer:
        shutil.copyfileobj(uploaded_file, buffer)


def initialize_session_state():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def make_qabot(uploaded_file_name: str, chunk_size: int, overlap_size: int):
    loader = PyMuPDFLoader(f"./temp/{uploaded_file_name}")
    pages = loader.load()
    st.write(f"Number of pages: {len(pages)} loaded!")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )
    splited_docs = text_splitter.split_documents(pages)

    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splited_docs,
        embedding=embedding,
        persist_directory="./docs/chroma",
    )
    vectordb.persist()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0.5, model="gpt-4")

    template = """..."""  # Your prompt template here
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    st.session_state["qa_chain"] = qa_chain


def main():
    st.title("QA APP")

    ensure_temp_directory()
    initialize_session_state()

    uploaded_file = st.file_uploader("Choose a PDF or text file", type="pdf")

    if uploaded_file is not None:
        save_uploaded_file(uploaded_file)

        st.info("""한글: 1000, 200  \n영어: 200, 40""")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("chunk_size:", key="chunk_size", value=200)
        with col2:
            overlap_size = st.number_input(
                "overlap_size:", key="overlap_size", value=40
            )

        if st.button("Submit"):
            make_qabot(uploaded_file.name, chunk_size, overlap_size)

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "question": prompt})
        full_response = st.session_state["qa_chain"]({"query": prompt})

        st.session_state.messages.append(
            {
                "role": "assistant",
                "answer": full_response["result"],
                "source_documents": full_response["source_documents"],
            }
        )

    if st.session_state["messages"]:
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                user_message = st.chat_message("user")
                user_message.write(message["question"])
            elif message["role"] == "assistant":
                assistant_message = st.chat_message("assistant")
                assistant_message.write(message["answer"])
                for source in message["source_documents"]:
                    assistant_message.write(source.page_content)


if __name__ == "__main__":
    main()
