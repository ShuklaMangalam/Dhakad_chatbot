from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="True"

prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please provide response to user queries"),
        ("user","Question:{question}")
    ]
)

st.title("Welcome to dhakad Chatbot")
input_text=st.text_input("Search the topic you want")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash")
output_parser=StrOutputParser()
chain = prompt_template|llm|output_parser
if input_text:
    st.write(chain.invoke({"question":input_text}))
