import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from the .env file
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

# upload PDF files

st.header("My first chatbot")

with st.sidebar:
    st.title("your documents")
    st.toast(api_key)
    file = st.file_uploader("upload a pdf file and start asking questions", type="pdf")

# extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text +=page.extract_text()

# break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=list("\n"),
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)
