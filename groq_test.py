import os
from langchain_groq import ChatGroq

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("GROQ_API_KEY not found in environment variables.")
else:
    try:
        llm = ChatGroq(model="llama-3-8b", api_key=api_key)
        response = llm.invoke("Hello, are you there?")
        print("Groq API response:", getattr(response, "content", None) or getattr(response, "text", None) or response)
    except Exception as e:
        print("Error connecting to Groq API:", e)
