import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI  # Usando a versão correta
from langchain_chroma import Chroma  # Usando a versão correta
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 
# Carregar variáveis de ambiente (para evitar expor a API Key no código)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
# 1. Carregar documentos PDF
print("📂 Carregando documentos...")
loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()
 
# Exibir quantidade de documentos carregados
print(f"✅ Documentos carregados: {len(documents)}")
 
# 2. Dividir os documentos em pedaços menores para melhorar a recuperação de contexto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
 
# Exibir quantidade de chunks criados
print(f"✅ Total de textos divididos (chunks): {len(texts)}")
 
# 3. Criar embeddings e indexar documentos com ChromaDB (verifica cache antes)
db_path = "./chroma_db"
 
# Criar embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
 
# Se o banco de dados já existir, carregue-o. Caso contrário, crie os embeddings.
if os.path.exists(db_path) and os.listdir(db_path):
    print("🔄 Base de conhecimento carregada do ChromaDB.")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    print("📂 Criando base de conhecimento no ChromaDB...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
 
    # Verificar se os documentos realmente foram adicionados
    num_docs = db._collection.count()
    if num_docs == 0:
        print("⚠️ Nenhum documento foi indexado! Verifique se os embeddings foram gerados corretamente.")
    else:
        print(f"✅ {num_docs} documentos foram indexados no ChromaDB.")
 
# Teste de busca para garantir que os documentos podem ser recuperados
test_query = "teste"
test_docs = db.similarity_search(test_query, k=3)
 
print("\n🔍 Teste de busca no ChromaDB:")
if not test_docs:
    print("⚠️ Nenhum documento encontrado! O ChromaDB pode estar vazio ou os embeddings não foram gerados corretamente.")
else:
    for i, doc in enumerate(test_docs):
        print(f"\n--- Documento {i+1} ---\n{doc.page_content[:500]}...\n")
 
# 4. Definir Prompt Personalizado para melhorar respostas
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Você é um assistente especializado nos arquivos da pasta docs. "
             "Use prioritariamente as informações desta pasta para responder de forma clara e objetiva:\n\n"
             "{context}\n\nPergunta: {question}\nResposta:"
)
 
# 5. Configurar o LLM e a cadeia de QA com o novo prompt
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
 
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)
 
# 6. Função para perguntar ao chatbot, exibindo documentos recuperados
def ask_question(question):
    retrieved_docs = db.similarity_search(question, k=3)  # Buscar os 3 documentos mais relevantes
 
    # Se nenhum documento relevante for encontrado, avisa o usuário
    if not retrieved_docs:
        return "Desculpe, não encontrei informações relevantes nos documentos."
 
    # Exibe os documentos recuperados no terminal
    print("\n🔍 Documentos Recuperados para esta Pergunta:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Documento {i+1} ---\n{doc.page_content}\n")
 
    # Passa apenas os documentos recuperados para o LLM
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    result = qa_chain({"query": question, "context": context})
   
    return result["result"]
 
# 7. Executar o chatbot no terminal
if __name__ == "__main__":
    print("🔹 Chatbot RAG ativo! Pergunte algo ou digite 'sair' para encerrar.")
    while True:
        question = input("\nVocê: ")
        if question.lower() in ["sair", "exit"]:
            print("🔹 Chatbot encerrado. Até mais!")
            break
        response = ask_question(question)
        print(f"🤖 Chatbot: {response}")