import streamlit as st
import os
from PIL import Image
from main import get_llm_response

def save_uploaded_file(uploaded_file):
    dir_name = "uploaded_images"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    try:
        with open(os.path.join(dir_name, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        return os.path.join(dir_name, uploaded_file.name)
    except Exception as e:
        print(e)
        return None
    
# Setting page title and header
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("AIDO")

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("Gemini-Pro", "Gemini-Pro-Vision"))

clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "Gemini-Pro":
    model = "gemini-pro"
else:
    model = "gemini-pro-vision"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []

# generate a response
def generate_response(prompt,model,image_path=None):
    st.session_state['past'].append(prompt)
    response = get_llm_response(prompt, model,image_path)
    st.session_state['generated'].append(response)
    return response

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        submit_button = st.form_submit_button(label='Send')
    
    if uploaded_file is not None:
        image_path = save_uploaded_file(uploaded_file)
        if image_path is not None:
            output = generate_response(user_input,model,image_path)   
    elif submit_button and user_input:
        output = generate_response(user_input,model)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            st.write(st.session_state["past"][i])
            st.write(output)
