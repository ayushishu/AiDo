import streamlit as st
from main import get_llm_response

st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("AIDO")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input,image=False))