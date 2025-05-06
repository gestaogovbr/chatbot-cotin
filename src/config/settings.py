import os
from dotenv import load_dotenv

def load_config():
    """Carrega variáveis de ambiente e configurações."""
    load_dotenv()

    config = {
        "databricks_host": os.getenv("DATABRICKS_HOST"),
        "databricks_token": os.getenv("DATABRICKS_TOKEN"),
        "db_path": "./chroma_db",
        "docs_dir": "docs",
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "similarity_threshold": 0.65,
        "max_tokens": 4096,
        "temperature": 0,
        "keywords": ["api", "módulos", "pncp", "painel", "catmat", "catser", "compras", "transparência"],
        "csv_options": {
            "default_delimiter": ",",
            "default_quotechar": '"',
            "encoding": "utf-8",
            "handle_headers": True,
        }
    }

    return config