import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def normalize_text(text: str) -> str:
    """Normaliza o texto removendo espaços extras."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def process_documents(documents):
    """Processa e normaliza o conteúdo dos documentos."""
    for i, doc in enumerate(documents):
        if not doc.page_content:
            doc.page_content = ""
            continue

        # Processamento específico por tipo de documento
        source_type = doc.metadata.get("source_type", "")

        if source_type == "csv":
            # Processamento específico para CSV
            if doc.page_content.startswith('{') and doc.page_content.endswith('}'):
                try:
                    import json
                    data = json.loads(doc.page_content)
                    # Formata como pares chave-valor
                    doc.page_content = "\n".join([f"{k}: {v}" for k, v in data.items()])
                except:
                    # Se não for JSON válido, mantém como está
                    pass
        elif source_type == "xlsx":
            # Processamento específico para Excel
            # Pode adicionar formatação específica para dados tabulares
            pass
        elif source_type == "db":
            # Processamento específico para dados de banco de dados
            # Pode formatar resultados de consultas SQL
            pass

        # Normalização comum para todos os tipos de documentos
        doc.page_content = normalize_text(doc.page_content)

        # Log de depuração
        print(f"[DEBUG] Amostra do documento {i+1} ({source_type}): {doc.page_content[:500]}")

    return documents

def split_documents(documents, chunk_size=1500, chunk_overlap=200):
    """Divide os documentos em chunks menores."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Total de textos (chunks): {len(texts)}")
    return texts

def process_csv_document(doc):
    """Processamento específico para documentos CSV."""
    # Verifica se é um documento CSV
    if doc.metadata.get("source_type") == "csv":
        # Formata o conteúdo para ser mais legível e útil para o RAG
        content = doc.page_content

        # Se o conteúdo for um dicionário ou objeto JSON, formata-o melhor
        if content.startswith('{') and content.endswith('}'):
            import json
            try:
                data = json.loads(content)
                # Formata como pares chave-valor
                formatted_content = "\n".join([f"{k}: {v}" for k, v in data.items()])
                doc.page_content = formatted_content
            except:
                # Se não for JSON válido, mantém como está
                pass

        # Normaliza o texto
        doc.page_content = normalize_text(doc.page_content)

    return doc