import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

st.header("Research Assistant")
user_input=st.text_input("Enter yoru research query:")
model=ChatOpenAI(model='gpt-4o-mini')
if st.button("Summarize"):
    result=model.invoke(user_input)
    st.write(result.content)