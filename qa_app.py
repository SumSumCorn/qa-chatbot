import streamlit as st
import os
import shutil
import fitz
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()

st.title("QA APP")


# Ensure the temp folder exists
if not os.path.exists("./temp"):
    os.makedirs("./temp")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader
uploaded_file = st.file_uploader("Choose a PDF or text file", type="pdf")

if uploaded_file is not None:
    with open(f"./temp/{uploaded_file.name}", "wb") as buffer:
        # Write to the buffer
        shutil.copyfileobj(uploaded_file, buffer)

    st.write("한글: 1000, 200")
    st.write("영어: 200, 40")
    chunk_size = st.slider("chunk_size:", 0, 2000, 1000)
    overlap_size = st.slider("overlap_size:", 0, 1000, 200)

    if st.button("Submit", type="primary"):
        loader = PyMuPDFLoader(f"./temp/{uploaded_file.name}")
        pages = loader.load()

        if pages is not None:
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

            memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

            retriever = vectordb.as_retriever()
            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0),
                retriever=retriever,
                memory=memory,
            )
            st.session_state["qa"] = qa

    if prompt := st.chat_input("What is up?"):
        qa = st.session_state["qa"]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = qa({"question": prompt})
            message_placeholder.markdown(full_response)

            # message_placeholder.markdown(full_response["source_documents"])

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response["answer"]}
            )


st.session_state
