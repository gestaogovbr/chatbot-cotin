import os
import re
import warnings
import numpy as np
from typing import List
import chainlit as cl  # <-- Importamos Chainlit

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Carregadores e splitters
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Modelos e embeddings Databricks
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings

# VectorStore
from langchain_community.vectorstores import Chroma

# Prompt e Chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Memória para histórico
from langchain.memory import ConversationBufferWindowMemory

warnings.filterwarnings('ignore')

# ================================
# Função para normalizar texto
# ================================
def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# ================================
# 1) Carrega variáveis de ambiente
# ================================
load_dotenv()
databricks_host = os.getenv("DATABRICKS_HOST")
databricks_token = os.getenv("DATABRICKS_TOKEN")

# ================================
# 2) Carrega documentos PDF e DOCX
# ================================
print("Carregando documentos...")

pdf_loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
pdf_documents = pdf_loader.load()

docx_loader = DirectoryLoader("docs", glob="**/*.docx", loader_cls=Docx2txtLoader)
docx_documents = docx_loader.load()

documents = pdf_documents + docx_documents
if not documents:
    raise ValueError("Nenhum documento (PDF ou DOCX) foi carregado na pasta 'docs'.")

print(f"Documentos carregados: {len(documents)} (PDFs: {len(pdf_documents)}, DOCX: {len(docx_documents)})")

for i, doc in enumerate(documents):
    if doc.page_content:
        doc.page_content = normalize_text(doc.page_content)
        print(f"[DEBUG] Amostra do documento {i+1}: {doc.page_content[:500]}")
    else:
        doc.page_content = ""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Total de textos (chunks): {len(texts)}")

# ================================
# 3) Configura embeddings e função filter_relevant_documents
# ================================
db_path = "./chroma_db"
embeddings = DatabricksEmbeddings(
    host=databricks_host,
    api_token=databricks_token,
    endpoint="databricks-bge-large-en"
)

def filter_relevant_documents(question: str, documents: List) -> List:
    question_embedding = np.array(embeddings.embed_query(question)).reshape(1, -1)
    doc_embeddings = np.array([
        np.array(embeddings.embed_query(doc.page_content))
        for doc in documents
    ])
    similarities = cosine_similarity(question_embedding, doc_embeddings)[0]
    
    keywords = ["api", "módulos", "pncp", "painel", "catmat", "catser", "compras", "transparência"]
    question_lower = question.lower()
    filtered_docs = []
    for doc, sim in zip(documents, similarities):
        doc_lower = doc.page_content.lower()
        keyword_score = sum(1 for kw in keywords if kw in doc_lower)
        final_score = sim + (keyword_score * 0.1)
        if final_score >= 0.65:
            filtered_docs.append((doc, final_score))
    
    filtered_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)
    relevant_docs = [doc for doc, _ in filtered_docs[:5]]
    return relevant_docs if relevant_docs else documents[:5]

# ================================
# 4) PromptTemplate e memória
# ================================
template = """
{% if chat_history %}
**Histórico da conversa (últimas interações):**
{{ chat_history }}
{% endif %}

1. Você é o Cotin IA
   Uma versão de Inteligência Artificial da Coordenação de Transparência e Informações Gerenciais, 
   especializada em dados abertos sobre compras públicas. 
   Responde com precisão e objetividade, sempre baseada em normativos do Compras.Gov, 
   legislação sobre licitações no Brasil e Lei de Acesso à Informação (Lei nº 12.527/2011). 
   Fala na primeira pessoa, com tom direto e eficiente, sem tolerar preguiça ou falta de esforço. 
   As respostas são extremamente estruturadas e fundamentadas nos normativos vigentes.

2. Instruções e Restrições
   • SEMPRE siga as regras definidas neste prompt.
   • SEMPRE responda no mesmo idioma da pergunta.
   • SEMPRE priorize as informações dos documentos armazenados na base de conhecimento ('docs').
   • SEMPRE forneça a URL exata quando fizer referência a uma fonte permitida.
   • NUNCA responda perguntas fora do foco de transparência, dados abertos e licitações; se ocorrer, retorne ao contexto.
   • NUNCA forneça informações fora das normativas do Compras.Gov ou LAI.
   • NUNCA utilize termos ou expressões da blacklist.

Você é o Cotin IA, assistente especializado em dados abertos sobre compras públicas. 
Responda à pergunta abaixo ({{ question }}) usando as informações extraídas da base de conhecimento (contexto). 
Se não houver informações suficientes nos documentos, forneça uma resposta baseada no meu conhecimento interno sobre a API de Compras.

Informações relevantes encontradas nos documentos:
{{ context }}

Pergunta do Usuário:
{{ question }}

Resposta:
"""

prompt_template = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template_format="jinja2",
    template=template,
)

memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    input_key="question"
)

llm = ChatDatabricks(
    host=databricks_host,
    api_token=databricks_token,
    endpoint="databricks-dbrx-instruct",
    max_tokens=3000,  # Ajuste conforme necessário
    temperature=0,
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=False
)

# ================================
# 5) ask_question e inicialização do Chroma
# ================================
def ask_question(question: str, retriever) -> str:
    docs = retriever.get_relevant_documents(question)
    print(f"[DEBUG] Documentos brutos recuperados: {len(docs)}")

    relevant_docs = filter_relevant_documents(question, docs)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    print(f"[DEBUG] Documentos filtrados (primeiros 1000 caracteres):\n{context[:1000]}\n")
    
    result = llm_chain({
        "question": question,
        "context": context,
    })
    return result["text"]


print("Inicializando/Carregando ChromaDB...")
if os.path.exists(db_path) and os.listdir(db_path):
    print("Carregando base de conhecimento do ChromaDB...")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    print("Criando base de conhecimento no ChromaDB...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    print(f"{db._collection.count()} documentos indexados no ChromaDB.")

retriever = db.as_retriever(search_kwargs={"k": 30})
print("Base de conhecimento pronta.")

# ================================
# 6) Integração com Chainlit
# ================================
@cl.on_chat_start
async def start_chat():
    """
    Quando o chat iniciar, enviamos uma mensagem de boas-vindas no UI.
    Também podemos zerar a memória, se quisermos.
    """
    memory.clear()
    await cl.Message(
        content="Olá! Sou o Cotin IA, pronto para ajudar com dados abertos de compras públicas.\nDigite sua pergunta!"
    ).send()

@cl.on_message
async def main(msg):
    """
    Recebe a mensagem do usuário (msg), extrai o texto e chama ask_question.
    Retorna a resposta via Chainlit UI.
    """
    user_text = msg.content if hasattr(msg, "content") else str(msg)
    if not isinstance(user_text, str):
        user_text = str(user_text)
    
    # 1) Cria uma mensagem "placeholder" para sinalizar que o bot está pensando.
    placeholder = await cl.Message(content="Processando, por favor aguarde...").send()

    # 2) Gera a resposta pelo LLM
    resposta = ask_question(user_text, retriever)

    # 3) Atualiza a mensagem de placeholder com o conteúdo final
    placeholder.content = resposta
    await placeholder.update()
