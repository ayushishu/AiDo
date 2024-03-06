from langchain_community.document_loaders import DirectoryLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from fastapi import FastAPI
import uuid
import uvicorn

chat_history = []

def load_image_llm():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDSVg488Lh7drFAadJcvgLYdZ4N1erjQnw"
    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision",convert_system_message_to_human=True)
    return llm

def load_text_llm():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDSVg488Lh7drFAadJcvgLYdZ4N1erjQnw"
    llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)
    return llm

def save_string_to_txt(input_string, directory="C:/Users/katoc/Downloads/node-audio-getter/uploadFile"):
    # Generate a unique filename using UUID
    unique_filename = str(uuid.uuid4()) + ".txt"

    # Combine the directory path and the unique filename
    filepath = os.path.join(directory, unique_filename)

    try:
        # Open the file in write mode
        with open(filepath, "w") as file:
            # Write the input string to the file
            file.write(input_string)
        print(f"String successfully saved to {filepath}")
    except Exception as e:
        print(f"Error: {e}")

def get_result(vectordb,query,model,image_path=None):
    
    if(model=="gemini-pro-vision"):
        
        image_llm = load_image_llm()
        message = HumanMessage(
        content=[
                {
                    "type": "text",
                    "text": query,
                },
                {"type": "image_url", "image_url": image_path},
            ]
        )
        content = image_llm.invoke([message]).content
        # save_string_to_txt(content)
        return content
    else:
        text_llm = load_text_llm()

        retriever = vectordb.as_retriever()

        # prompt = hub.pull("rlm/rag-prompt")

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        contextualize_q_chain = contextualize_q_prompt | text_llm | StrOutputParser()

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | text_llm
        )

        if(query.find("summary")!=-1 or query.find("Summary")!=-1 or query.find("summarize")!=-1 or query.find("Summarize")!=-1) :
            summarize_chain = load_summarize_chain(text_llm,chain_type="stuff")
            search = vectordb.similarity_search(" ")
            summary = summarize_chain.invoke(input_documents=search, question="Write a summary within 300 words.")
            return summary["output_text"]
        else:
            res = rag_chain.invoke({"question": query, "chat_history": chat_history})
            chat_history.extend([HumanMessage(content=query), res])
            return res.content
 
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def chroma_db_store(load_db):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_directory = "chroma_db"

    if(load_db):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

        return vectordb
    else:
        dir = "C:/Users/katoc/Downloads/node-audio-getter/uploadFile"

        documents = load_docs(dir)

        docs = split_docs(documents)

        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )

        vectordb.persist()
        return vectordb

def get_llm_response(query,model,image_path=None):

    vectordb = chroma_db_store(load_db=False)

    result = get_result(vectordb,query,model,image_path)

    return result



