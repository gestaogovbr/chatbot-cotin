# Cotin IA - Assistente de Transparência e Compras Públicas 🚀📊

Este projeto implementa o **Cotin IA**, uma versão de Inteligência Artificial da **Coordenação de Transparência e Informações Gerenciais - COTING/CGGES/DELOG**, especializada em **dados abertos sobre compras públicas**. O assistente responde com precisão e objetividade, sempre baseado em normativos do **Compras.Gov**, **Lei de Acesso à Informação (LAI - Lei nº 12.527/2011)** e outras regulamentações.

---

## 📌 Funcionalidades

- **Carregamento de Documentos**: Importa arquivos **PDF e DOCX** para criar uma base de conhecimento.
- **Processamento de Texto**: Normaliza e divide textos em **chunks** para melhor indexação.
- **Busca Semântica Inteligente**: Usa **Databricks Embeddings** e **ChromaDB** para recuperar documentos relevantes.
- **Modelo de IA para Respostas**: Utiliza **ChatDatabricks** para gerar respostas estruturadas e baseadas em normativos.
- **Memória de Conversação**: Mantém o histórico de interações para fornecer respostas mais contextuais.
- **Integração com Chainlit**: Interface interativa para perguntas e respostas em tempo real.

---

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Chainlit** (Interface Conversacional)
- **Databricks Embeddings**
- **LangChain**
- **ChromaDB** (Banco de vetores)
- **scikit-learn (cosine similarity)**
- **dotenv (para variáveis de ambiente)**

---

## 🚀 Como Executar

1️⃣ Instale as Dependências
```sh
# Criar venv
pip install -r requirements.txt

2️⃣ Configure as Variáveis de Ambiente
Crie um arquivo .env na raiz do projeto e adicione suas credenciais do Databricks:

DATABRICKS_HOST=SEU_HOST
DATABRICKS_TOKEN=SEU_TOKEN

3️⃣ Coloque os Documentos na Pasta docs
Certifique-se de que os arquivos PDF e DOCX que deseja processar estão dentro do diretório docs/.

4️⃣ Execute o Servidor Chainlit
chainlit run main.py

Após isso, acesse o chat interativo e comece a fazer perguntas!