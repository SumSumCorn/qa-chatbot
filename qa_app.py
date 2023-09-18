import streamlit as st
import numpy as np


st.title("QA APP ")

if st.file_uploader('chosse a pdf or text file',accept_multiple_files=True):
    if st.button('submit',type='primary'):
        

    
# if "messages" not in st.session_state:
#     st.session_state.messages = []


# for message in st.session_state.messages:
#     with st.chat_message(message['role']):
#         st.markdown(message["content"])



# if prompt := st.chat_input("Say Something"):
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     st.session_state.messages.append({"role": "user", "content": prompt})

#     response = f"Echo: {prompt}"

#     with st.chat_message("assistant"):
#         st.markdown(response)

#     st.session_state.messages.append({"role": "assistant", "content": response})
