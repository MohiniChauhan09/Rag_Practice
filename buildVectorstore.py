from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from typing import List

from utils.pdf_loader import load_pdfs_from_dir 
from config import ( DATA_UPLOADS, DB_FAISS_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP )

class SimpleCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        out: List[Document] = []
        for doc in documents:
            text = getattr(doc, "page_content", str(doc)) or ""
            start = 0
            L = len(text)
            while start < L:
                end = min(start + self.chunk_size, L)
                chunk = text[start:end]
                out.append(Document(page_content=chunk, metadata=getattr(doc, "metadata", {})))
                start = max(end - self.chunk_overlap, end)
        return out

class LocalSentenceEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "pip install sentence-transformers\n"
                "If you see torch-related delays, install CPU wheel:\n"
                "pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --no-deps"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

print("Loading Class 12 NCERT PDFs...") 
docs = load_pdfs_from_dir(DATA_UPLOADS) 
if not docs:
    raise RuntimeError("No PDFs found in data/uploads/class12") 

print("Splitting content...") 
splitter = SimpleCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) 
chunks = splitter.split_documents(docs) 

print("Creating embeddings...") 
embeddings = LocalSentenceEmbeddings(EMBED_MODEL) 

print("Building FAISS vectorstore...") 
db = FAISS.from_documents(chunks, embeddings) 
db.save_local(DB_FAISS_PATH) 
print("Vectorstore created successfully!")