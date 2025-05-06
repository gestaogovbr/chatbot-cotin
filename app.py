import warnings
import chainlit as cl

# Importações dos módulos refatorados
from src.config.settings import load_config
from src.loaders.document_loaders import load_all_documents
from src.processing.text_processor import process_documents, split_documents
from src.retrieval.embeddings import get_embeddings
from src.retrieval.vector_store import initialize_vector_store
from src.llm.prompts import get_prompt_template
from src.llm.chain import get_memory, get_llm, setup_chain, ask_question

# Suprimir avisos
warnings.filterwarnings('ignore')

# Carregar configurações
config = load_config()

# Carregar e processar documentos
documents = load_all_documents(config["docs_dir"])
processed_docs = process_documents(documents)
texts = split_documents(
    processed_docs,
    config["chunk_size"],
    config["chunk_overlap"]
)

# Configurar embeddings
embeddings = get_embeddings(
    config["databricks_host"],
    config["databricks_token"]
)

# Inicializar vector store
db = initialize_vector_store(texts, embeddings, config["db_path"])
retriever = db.as_retriever(search_kwargs={"k": len(documents)})

# Configurar LLM e chain
prompt_template = get_prompt_template()
memory = get_memory()
llm = get_llm(
    config["databricks_host"],
    config["databricks_token"],
    config["max_tokens"],
    config["temperature"]
)
llm_chain = setup_chain(llm, prompt_template, memory)

# Configurar handlers do Chainlit
@cl.on_chat_start
async def on_chat_start():
    # Armazena funções e objetos na sessão do usuário para uso nos callbacks
    cl.user_session.set("ask_question_func", ask_question)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("llm_chain", llm_chain)
    cl.user_session.set("embeddings", embeddings)
    cl.user_session.set("keywords", config["keywords"])
    cl.user_session.set("threshold", config["similarity_threshold"])
    cl.user_session.set("user_id", "anônimo")  # Você pode implementar identificação de usuário se necessário

    # Importa e inicia o chat
    from src.ui.chainlit_handlers import start_chat
    await start_chat(memory)

@cl.on_message
async def on_message(msg):
    # Importa e processa a mensagem
    from src.ui.chainlit_handlers import handle_message
    await handle_message(
        msg,
        ask_question,
        retriever,
        llm_chain,
        embeddings,
        config["keywords"],
        config["similarity_threshold"]
    )