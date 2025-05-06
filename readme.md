# Cotin IA - Sistema RAG para Transparência em Compras Públicas 🚀📊

## 📑 Visão Geral

O **Cotin IA** é um sistema avançado de Recuperação Aumentada por Geração (RAG) desenvolvido pela **Coordenação de Transparência e Informações Gerenciais — COTIN/CGGES/DELOG**. Especializado em dados abertos sobre compras públicas, o assistente fornece respostas precisas e objetivas, fundamentadas nos normativos do **Compras.Gov**, **Lei de Acesso à Informação (LAI — Lei nº 12.527/2011)** e outras regulamentações pertinentes.

---

## 🌟 Funcionalidades Principais

* **Processamento multi‑formato**: importa e processa arquivos **PDF, DOCX, XLSX, CSV e bancos de dados SQLite**.
* **Processamento avançado de texto**: normaliza e divide textos em *chunks* otimizados para indexação e recuperação eficiente.
* **Busca semântica inteligente**: utiliza *embeddings* vetoriais e similaridade de cosseno.
* **Filtragem por relevância**: sistema de pontuação que combina similaridade semântica e presença de palavras‑chave.
* **Respostas contextuais**: gera respostas baseadas nos documentos recuperados e no histórico de conversa.
* **Memória de conversação**: mantém o histórico recente para maior contextualização.
* **Interface interativa**: integração com Chainlit para uma experiência de chat intuitiva.

---

## 🏗️ Arquitetura Modular

```text
project/
├── app.py               # Ponto de entrada principal
├── requirements.txt     # Dependências do projeto
├── .env                 # Variáveis de ambiente
├── chainlit.md          # Documentação da interface
├── docs/                # Documentos para processamento
├── chroma_db/           # Banco de dados vetorial
└── src/                 # Código‑fonte organizado
    ├── loaders/         # Carregadores de documentos
    ├── processing/      # Processamento de texto
    ├── retrieval/       # Recuperação de documentos
    ├── llm/             # Interação com LLM
    ├── config/          # Configurações
    └── ui/              # Interface com usuario
```

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.9+**
* **LangChain**
* **Databricks Embeddings**
* **ChatDatabricks**
* **ChromaDB**
* **Chainlit**
* **scikit‑learn** (similaridade de cosseno)
* **PyMuPDF** & **Docx2txt**
* **Pandas** & **Openpyxl**
* **SQLite**

---

## 📋 Pré‑requisitos

1. Python **3.9** ou superior
2. Acesso à plataforma **Databricks** (para embeddings e LLM)
3. Pelo menos **4 GB** de RAM disponível
4. Espaço em disco suficiente para armazenar documentos e o banco vetorial

---

## 🚀 Guia de Instalação e Execução

### 1. Clone o repositório

```bash
git clone https://github.com/gestaogovbr/chatbot-cotin.git
cd chatbot-cotin.git
```

### 2. Crie e ative um ambiente virtual

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Crie o arquivo **.env** na raiz do projeto:

```env
DATABRICKS_HOST=seu-host-databricks
DATABRICKS_TOKEN=seu-token-databricks
```

### 5. Adicione documentos à base de conhecimento

```bash
mkdir -p docs
# copie seus PDF, DOCX, XLSX e arquivos .db para a pasta docs/
```

### 6. Execute a aplicação

```bash
chainlit run app.py
```

Acesse **[http://localhost:8000](http://localhost:8000)** e comece a fazer perguntas!

---

## 🔄 Utilizando Modelos Alternativos

O Cotin IA é flexível e suporta diversos provedores de *embeddings* e LLMs.

### OpenAI

```python
# src/config/settings.py
config = {
    # ...
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
}

# src/retrieval/embeddings.py
from langchain_openai import OpenAIEmbeddings

def get_embeddings(config):
    return OpenAIEmbeddings(
        api_key=config["openai_api_key"],
        model="text-embedding-3-small",
    )

# src/llm/chain.py
from langchain_openai import ChatOpenAI

def get_llm(config):
    return ChatOpenAI(
        api_key=config["openai_api_key"],
        model_name="gpt-4o",
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
```

### Modelos locais (Ollama)

```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

def get_embeddings(config):
    return OllamaEmbeddings(model="nomic-embed-text")

def get_llm(config):
    return Ollama(
        model="llama3",
        temperature=config["temperature"],
    )
```

### HuggingFace

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

def get_embeddings(config):
    return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def get_llm(config):
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=config["hf_token"],
        max_new_tokens=config["max_tokens"],
        temperature=config["temperature"],
    )
```

---

## 🤝 Contribuições

1. Faça um **fork** do projeto.

2. Crie uma *branch* para sua feature:

   ```bash
   git checkout -b feature/minha-feature
   ```

3. *Commit* das mudanças:

   ```bash
   git commit -m "feat: adiciona minha-feature"
   ```

4. *Push* para o seu fork:

   ```bash
   git push origin feature/minha-feature
   ```

5. Abra um **Pull Request**.

---

## 📜 Licença

Distribuído sob a [licença MIT](LICENSE).

---

## 👥 Equipe

**Coordenação de Transparência e Informações Gerenciais — COTIN**

* **Coordenador:** Magnum Costa de Oliveira
* **Equipe:**

  * Guilherme Fonseca De Noronha Rocha
  * Stefano Terci Gasperazzo
  * Jose Maria De Melo Junior
  * Luiz Gonzaga de Oliveira
  * André Ruperto de Macêdo
