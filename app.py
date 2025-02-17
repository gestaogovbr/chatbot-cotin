import os
from dotenv import load_dotenv
# Carregadores, splitters, embeddings etc. (OK usar as versões antigas se funcionam)
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma

# Import do LLMChain e PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 1. Carrega variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. Carrega e indexa documentos
print("Carregando documentos...")
loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()
print(f"Documentos carregados: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"Total de textos (chunks): {len(texts)}")

db_path = "./chroma_db"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if os.path.exists(db_path) and os.listdir(db_path):
    print("Carregando base de conhecimento do ChromaDB...")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    print("Criando base de conhecimento no ChromaDB...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    print(f"{db._collection.count()} documentos indexados no ChromaDB.")

# 3. Cria retriever (para buscar documentos relevantes)
retriever = db.as_retriever()

# 4. Define o PromptTemplate com condicional (Geral x SIC)
template = """
Você é um assistente especializado em responder perguntas com base nos documentos armazenados na base de conhecimento (pasta 'docs').
Priorize exclusivamente as informações desses documentos para fornecer respostas precisas, concisas e baseadas em fatos.

{% if request_type == "SIC" %}
**Teor da Manifestação**
Resumo: {{ sic_summary }}
Extrato: {{ sic_extract }}

**Resposta no formato SIC:**

Prezado(a) Senhor(a),
1. Faço referência ao Pedido de Informação (SEI {{ sei_number }}), ...
   ...
Atenciosamente,
[Nome do Setor Responsável]

{% else %}
**Resposta Geral:**

Com base nas informações disponíveis na base de conhecimento:
{{ context }}

Pergunta: {{ question }}
Resposta:
{% endif %}
"""

prompt_template = PromptTemplate(
    input_variables=[
        "context",
        "question",
        "request_type",
        "sic_summary",
        "sic_extract",
        "sei_number",
        "api_info",
        "additional_resources",
    ],
    template_format="jinja2",
    template=template,
)

# 5. Cria o LLMChain simples (sem RetrievalQA)
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 6. Função para perguntar
def ask_question(question: str) -> str:
    # Passo 1: buscar documentos
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "Desculpe, não encontrei informações relevantes nos documentos."

    # Passo 2: concatenar conteúdo em 'context'
    context = "\n".join([doc.page_content for doc in docs])

    # Passo 3: ver se é pergunta SIC ou geral
    if "SIC" in question.upper():
        # Resposta em formato SIC
        return llm_chain.run(
            context=context,
            question=question,
            request_type="SIC",
            sic_summary="Resumo da solicitação",
            sic_extract="Extrato da solicitação",
            sei_number="12345",
            api_info="",               # se tiver algo, passe aqui
            additional_resources="",   # se tiver algo, passe aqui
        )
    else:
        # Resposta geral
        return llm_chain.run(
            context=context,
            question=question,
            request_type="Geral",
            sic_summary="",
            sic_extract="",
            sei_number="",
            api_info="",
            additional_resources="",
        )

# 7. Loop interativo
if __name__ == "__main__":
    print("Chatbot RAG ativo! Pergunte algo ou digite 'sair' para encerrar.")
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit"]:
            print("Chatbot encerrado. Até mais!")
            break

        resposta = ask_question(user_input)
        print(f"\nChatbot:\n{resposta}\n")
