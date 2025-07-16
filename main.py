from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


import os
import pinecone

load_dotenv()
pinecone_api_key=os.getenv("PINECONE_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "langchain-rag-google"

def format_document(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.title("RAG Document Chatbot with LangChain and Pinecone")
st.markdown("This is the demo of a RAG Chatbot with Langchain and Pinecone")

uploaded_files=st.file_uploader("upload a pdf file",type=["pdf"],accept_multiple_files=True)

if uploaded_files:
    st.write("processing the files")

    docs=[]
    for uploaded_file in uploaded_files:
        # save file to a temporary location
        with open(uploaded_file.name,"wb") as f:
            f.write(uploaded_file.getvalue())


            #load the pdf file
            loader=PyMuPDFLoader(uploaded_file.name)
            docs.extend(loader.load())
            
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
