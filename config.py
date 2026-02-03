import os 
from dotenv import load_dotenv 
load_dotenv()

# ===== API ===== 
print("GROQ_API_KEY exists:", os.getenv("GROQ_API_KEY") is not None)

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# ===== MODELS ===== 
LLM_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ===== PATHS ===== 
DATA_UPLOADS = "data/uploads/class12" 
DB_FAISS_PATH = "data/vectorstore/faiss_db" 

# ===== CHUNKING ===== 
CHUNK_SIZE = 800 
CHUNK_OVERLAP = 120

print("GROQ_API_KEY loaded:", GROQ_API_KEY)
