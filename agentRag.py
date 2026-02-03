from langchain_groq import ChatGroq 
from sentence_transformers import SentenceTransformer 
from langchain_community.vectorstores import FAISS 
from pydantic import SecretStr
from typing import List
from langchain.embeddings.base import Embeddings


from config import (GROQ_API_KEY, LLM_MODEL, EMBED_MODEL, DB_FAISS_PATH )

class MyEmbeddings(Embeddings): 
    def __init__(self, model_name: str): 
        self.model = SentenceTransformer(model_name) 
    def embed_documents(self, texts: List[str]) -> List[List[float]]: 
        return self.model.encode(texts, convert_to_tensor=False).tolist() 
    def embed_query(self, text: str) -> List[float]: 
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()   
    
class Class12NCERTAgent: 
    def __init__(self): 
        self.llm = ChatGroq( 
            model=LLM_MODEL, 
            api_key=SecretStr(GROQ_API_KEY) if GROQ_API_KEY is not None else None, 
            temperature=0.1, 
            max_tokens=600 
        ) 
        self.embeddings = MyEmbeddings(EMBED_MODEL) 
        self.vectorstore = FAISS.load_local( DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True ) 
    
    def ask(self, question: str) -> str:
        import logging
        try:
            docs = self.vectorstore.similarity_search(question, k=5)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f""" You are a Class 12 NCERT Science teacher. \
            STRICT RULES: - Answer ONLY from the given NCERT content - If answer is not present, \
            say: "This topic is not covered in the Class 12 NCERT Science textbooks." \
            - Use simple, clear student language - No extra facts NCERT CONTENT: {context} Question: {question} Answer: """

            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", None) or getattr(response, "text", None) or response
            return str(raw).strip()
        except Exception as e:
            logging.exception("Error in Class12NCERTAgent.ask:")
            return f"An error occurred: {str(e)}"
    




## context = "\n\n".join(d.page_content for d in docs) 

# Extracts only the text content from each retrieved document.
# Joins them into a single string, separated by blank lines.
# This combined text becomes the knowledge source for the LLM.

