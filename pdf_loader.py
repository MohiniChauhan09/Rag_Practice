import os 
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs_from_dir(folder_path): 
    documents = [] 
    for file in os.listdir(folder_path): 
        if file.endswith(".pdf"): 
            full_path = os.path.join(folder_path, file) 
            loader = PyPDFLoader(full_path) 
            docs = loader.load() 
            for doc in docs: 
                doc.metadata["source"] = file 
            documents.extend(docs) 
    return documents


# CONTINUE FROM STEP 11