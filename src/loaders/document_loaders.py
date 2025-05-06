import os
import sqlite3
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader, SQLDatabaseLoader
from langchain_community.document_loaders import CSVLoader

def load_pdf_documents(docs_dir="docs"):
    """Carrega documentos PDF do diretório especificado."""
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    print(f"PDFs carregados: {len(documents)}")
    return documents

def load_docx_documents(docs_dir="docs"):
    """Carrega documentos DOCX do diretório especificado."""
    loader = DirectoryLoader(docs_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents = loader.load()
    print(f"DOCXs carregados: {len(documents)}")
    return documents

def load_xlsx_documents(docs_dir="docs"):
    """Carrega documentos XLSX do diretório especificado."""
    loader = DirectoryLoader(docs_dir, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader)
    documents = loader.load()
    print(f"XLSXs carregados: {len(documents)}")
    return documents

def load_db_documents(docs_dir="docs"):
    """Carrega documentos de bancos de dados SQLite."""
    db_documents = []
    for file in os.listdir(docs_dir):
        if file.endswith(".db"):
            db_path = os.path.join(docs_dir, file)
            db_documents.extend(_load_single_db(db_path))
    print(f"DBs carregados: {len(db_documents)}")
    return db_documents

def _load_single_db(db_path, query=None):
    """Carrega dados de um único arquivo de banco de dados."""
    if not query:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()

        documents = []
        for table in tables:
            table_name = table[0]
            loader = SQLDatabaseLoader(
                db_path,
                f"SELECT * FROM {table_name}",
                page_content_columns=None
            )
            documents.extend(loader.load())
        return documents
    else:
        loader = SQLDatabaseLoader(db_path, query)
        return loader.load()

def load_csv_documents(docs_dir="docs"):
    """Carrega documentos CSV do diretório especificado."""
    import os
    
    csv_documents = []
    for file in os.listdir(docs_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(docs_dir, file)
            loader = CSVLoader(
                file_path=file_path,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': None  # Detecta automaticamente os cabeçalhos
                },
                source_column=None  # Usa todas as colunas
            )
            documents = loader.load()
            # Adiciona metadados para identificar a origem
            for doc in documents:
                doc.metadata["source_type"] = "csv"
                doc.metadata["filename"] = file
            
            csv_documents.extend(documents)
    
    print(f"CSVs carregados: {len(csv_documents)}")
    return csv_documents

def load_all_documents(docs_dir="docs"):
    """Carrega todos os tipos de documentos suportados."""
    print("Carregando documentos...")

    pdf_documents = load_pdf_documents(docs_dir)
    docx_documents = load_docx_documents(docs_dir)
    xlsx_documents = load_xlsx_documents(docs_dir)
    db_documents = load_db_documents(docs_dir)
    csv_documents = load_csv_documents(docs_dir)  # Nova linha

    documents = pdf_documents + docx_documents + xlsx_documents + db_documents + csv_documents  # Atualizado

    if not documents:
        raise ValueError("Nenhum documento foi carregado na pasta 'docs'.")

    print(f"Total de documentos carregados: {len(documents)} (PDFs: {len(pdf_documents)}, "
          f"DOCX: {len(docx_documents)}, XLSX: {len(xlsx_documents)}, "
          f"DB: {len(db_documents)}, CSV: {len(csv_documents)})")  # Atualizado

    return documents



