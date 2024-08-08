import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

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

    # get user question
    user_question = st.text_input("type your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        # define the LLM
        llm = ChatOpenAI(
            openai_api_key=api_key,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # output result
        chain = load_qa_chain(llm, chain_type='stuff')
        response = chain.run(input_documents=match,question=user_question)
        st.write(response)
