import streamlit as st 
from agents.agentRag import Class12NCERTAgent 

st.set_page_config(page_title="Class 12 NCERT Science Assistant") 
st.title("📘 Class 12 NCERT Science Assistant") 
st.caption("Answers strictly from Class 12 NCERT textbooks") 
agent = Class12NCERTAgent() 
question = st.text_input("Ask a question from Class 12 Science:") 
if question: 
    with st.spinner("Searching NCERT books..."): 
        answer = agent.ask(question) 
        st.write(answer)