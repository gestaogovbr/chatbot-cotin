import os
from langchain_community.vectorstores import Chroma

def initialize_vector_store(texts, embeddings, db_path="./chroma_db"):
    """Inicializa ou carrega o banco de dados vetorial."""
    print("Inicializando/Carregando ChromaDB...")

    if os.path.exists(db_path) and os.listdir(db_path):
        print("Carregando base de conhecimento do ChromaDB...")
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        print("Criando base de conhecimento no ChromaDB...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
        print(f"{db._collection.count()} documentos indexados no ChromaDB.")

    print("Base de conhecimento pronta.")
    return db