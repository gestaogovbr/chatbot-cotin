import os
import re
from dotenv import load_dotenv

# Carregadores e splitters
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Observação: Verifique se esse import corresponde à sua versão do LangChain
# Em versões mais novas, "OpenAI" e "ChatOpenAI" vêm de "langchain.chat_models"
# Se você estiver usando "langchain_openai", ajuste conforme necessário.
from langchain.chat_models import ChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# 1. Carrega variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. Carrega e indexa documentos
print("Carregando documentos...")
loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()
print(f"Documentos carregados: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"Total de textos (chunks): {len(texts)}")

db_path = "./chroma_db"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Verifica se o diretório db_path existe e não está vazio
if os.path.exists(db_path) and os.listdir(db_path):
    print("Carregando base de conhecimento do ChromaDB...")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    print("Criando base de conhecimento no ChromaDB...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
    print(f"{db._collection.count()} documentos indexados no ChromaDB.")

# 3. Cria retriever
retriever = db.as_retriever(
    search_kwargs={"k": 5}  # Aumente o número de documentos retornados
)

# 4. Define o PromptTemplate (SIC/Geral)
template = """
Você é um assistente especializado em responder perguntas com base nos documentos armazenados na base de conhecimento (pasta 'docs').
Priorize as informações desses documentos para fornecer respostas precisas, concisas e baseadas em fatos. Se a resposta for no formato SIC, 
utilize o modelo request_type == "SIC" de forma completa (itens de 1 a 7) e busque informações na pasta 'docs' para te ajudar na personalização da resposta, 
substituindo conforme input que receber os trechos sei_number, sic_summary, sic_extract e context. Caso contrário, responda de forma geral.

{% if request_type == "SIC" %}
Prezado(a) Senhor(a),
A seguir, apresentamos a resposta completa ao seu pedido de informação, conforme o formato padrão SIC. **Não omita nenhum item do template, mesmo que algumas informações sejam genéricas ou repetitivas.**

1. Faço referência ao Pedido de Informação (SEI {{ sei_number }}), que em resumo trata sobre ({{ sic_summary }}), e no extrato o requerente assim se manifesta: "{{ sic_extract }}"

2. Inicialmente, cumpre informar que a Lei de Acesso à Informação (Lei nº 12.527/2011)
garante o direito de acesso a informações públicas e determina, em seu artigo 8º, inciso III, que os órgãos e
entidades públicas devem "possibilitar o acesso automatizado por sistemas externos em formatos abertos, 
estruturados e legíveis por máquina." Essa exigência visa assegurar que os dados divulgados sejam
acessíveis de forma direta e clara, permitindo que sistemas externos possam processá-los e reutilizá-los
para fins de análise, pesquisa ou controle social. O objetivo é promover a transparência e facilitar o 
acompanhamento das ações governamentais, garantindo que as informações sejam facilmente
manipuláveis por softwares, estimulando assim a inovação e a eficiência no uso de dados públicos.
Ademais, conforme § 3º, art. 11, da Lei de Acesso à Informação, sem prejuízo da segurança e da proteção
das informações e do cumprimento da legislação aplicável, o órgão ou entidade poderá oferecer meios
para que o próprio requerente possa pesquisar a informação de que necessitar.

3. Nesse sentido, especificamente em relação à demanda do requerente, seguem as informações relevantes encontradas na base de conhecimento:
{{ context }}

4. A API de Compras disponibiliza ampla quantidade de recursos em dados abertos relacionados às compras 
públicas do governo federal, que podem ser acessados por meio do Link:
(https://dados.gov.br/dados/conjuntos-dados/compras-publicas-do-governo-federal) ou
(https://dadosabertos.compras.gov.br/swagger-ui/index.html). 

A API contém atualmente 9 (nove) módulos
principais que cobrem diferentes aspectos das compras públicas:
4.1. Módulo 1: Catálogo de Materiais (CATMAT): Permite a consulta de itens de materiais, com
dados detalhados como quantidade, preço unitário, fabricante e marca.
4.2. Módulo 2: Catálogo de Serviços (CATSER): Permite consultar os serviços com informações
detalhadas.
4.3. Módulo 3: Preços Praticados: Permite consultar os preços praticados nas compras públicas,
utilizando os códigos dos itens do CATMAT e CATSER.
4.4. Módulo 4: Planejamento e Gerenciamento das Contratações (PGC): Oferece dados sobre o
planejamento e gerenciamento das contratações públicas, permitindo uma visão completa das aquisições
programadas pelos órgãos públicos.
4.5. Módulo 5: Unidades Administrativas de Serviços Gerais (UASG): Traz informações
detalhadas sobre as UASGs, como os dados da unidade, órgão superior, e a vinculação administrativa das
unidades responsáveis pela condução das licitações e contratações. Isso facilita a análise das entidades
envolvidas nas compras governamentais.
4.6. Módulo 6: Legado: Permite a consulta das compras e contratações realizadas sob a vigência
das leis anteriores à Lei nº 14.133/2021, como a Lei nº 8.666/1993. Esse módulo possibilita a análise de
processos licitatórios que ocorreram antes da nova legislação de licitações e contratos.
4.7. Módulo 7: Contratações (Lei nº 14.133/2021): Este módulo é focado em compras e
contratações realizadas sob a nova Lei nº 14.133/2021. Ele permite consultar os processos licitatórios e
contratações que estão de acordo com a legislação atual, trazendo transparência sobre os itens licitados, as
empresas participantes e os resultados das licitações, com informações detalhadas sobre os valores e as
etapas de cada processo.
4.8. Módulo 8: ARP - Ata de Registro de Preço: Possibilita a consulta de Atas de Registro de Preço
e seus respectivos itens, e traz informações detalhadas como por exemplo: unidade gerenciadora, órgão,
objeto, valor, descrição de itens, data de vigência inicial e final.
4.9. Módulo 9: Contratos: Permite a consulta de contratos e seus itens, e traz informações
detalhadas como por exemplo: nome do órgão, número do contrato, objeto, valor, descrição de itens,
nome do fornecedor, data de vigência inicial e final e descrição de itens.
4.10. Esses módulos permitem o acesso detalhado às informações de homologações e licitações,
facilitando a análise de dados por meio de planilhas ou dashboards. A documentação da API pode ser
acessada diretamente no seguinte link: (https://dadosabertos.compras.gov.br/swagger-ui/index.html).

5. Ademais, no intuito de contribuir com a busca de informações em dados abertos sobre
Compras Públicas, recomendamos outros Painéis e Portais:
5.1. Painel de Compras Governamentais: Recomendamos o uso do Painel de Compras, que
permite uma consulta rápida e detalhada das licitações realizadas, com filtros por data, órgão, e outros
parâmetros específicos. O painel facilita o acesso aos dados de homologação e permite exportar as
informações para planilhas. Link para consulta: https://paineldecompras.economia.gov.br/licitacao-sessao
5.2. Consulta Detalhada no Portal de Compras Governamentais: Para acessar diretamente os
dados de licitações e homologações, incluindo o número da licitação, órgão licitador, itens e empresas
vencedoras, sugerimos utilizar o seguinte link específico para consulta detalhada de licitações: Link para
consulta: https://www.gov.br/compras/pt-br/acesso-a-informacao/consulta-detalhada
5.3. Portal Nacional de Contratações Públicas (PNCP): O PNCP oferece consultas centralizadas
sobre contratações públicas, permitindo o acesso a dados de licitações e homologações, incluindo itens e
valores homologados. Esse portal oferece uma visão ampla das contratações públicas em âmbito municipal,
estadual e federal, além de disponibilizar informações sobre os contratos vigentes. Link para consulta:
https://pncp.gov.br/app/
5.4. Painel PNCP em números: painel interativo que mostra o ecossistema de compras públicas
em todo país, a nível municipal, estadual e federal.
Link: https://www.gov.br/pncp/pt-br/acesso-a-informacao/painel-pncp-em-numeros

6. Cabe ainda informar que é possível realizar buscas na transparência ativa do Catálogo do
Compras.gov.br, a partir da descrição desejada, para obtenção dos códigos de grupo e serviços ou classe e
código de materiais. A partir da obtenção dos códigos desejados, é possível utilizar o assistente do Pesquisa
de Preços, que é uma ferramenta do Compras.gov.br que permite aos usuários consultar os preços de
compras realizadas por meio do sistema:
6.1. Catálogo: https://catalogo.compras.gov.br/cnbs-web/busca
6.2. Pesquisa de Preços: https://www.gov.br/compras/pt-br/sistemas/conheca-o-compras/pesquisa-de-precos

7. Por fim, informa-se que é possível encontrar dados sobre contratos por meio do seguinte
caminho:
7.1. Contratos.gov.br: repositório contendo informações detalhadas sobre os empenhos e
contratos, a partir de 2021, bem como um dicionário de dados que fornece esclarecimentos específicos
sobre as informações disponíveis, estando nele as informações relativas aos itens, com a descrição dos seus
objetos. Link: https://dados.gov.br/dados/conjuntos-dados/comprasgovbr-contratos

8. O Governo Federal está comprometido em promover a transparência dos gastos públicos,
fornecer informações de valor agregado à sociedade e promover a pesquisa e inovação tecnológica por
meio da implementação da política brasileira de dados abertos.

{% else %}
**Resposta Geral:**

Com base nas informações disponíveis na base de conhecimento, seguem as informações relevantes para a sua pergunta:
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

# 5. Use ChatOpenAI em vez de OpenAI
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4",  # Exige ter acesso ao GPT-4
    temperature=0,
    max_tokens=3000,
)

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def parse_sic_input(user_input: str):
    """
    Captura:
    - Número do pedido (ex.: 123456, SEI 123456, Pedido 123456)
    - Resumo: entre 'Resumo:' e 'Extrato:'
    - Extrato: após 'Extrato:'
    Retorna (sei_number, sic_summary, sic_extract).
    Se não encontrar nada, retorna strings vazias.
    """
    # Captura o número do pedido (aceita SEI, Pedido, ou número entre parênteses)
    sei_match = re.search(r"(SEI|Pedido)\s*(\d+)|\((\d+)\)", user_input, re.IGNORECASE)
    sei_number = sei_match.group(2) or sei_match.group(3) if sei_match else ""

    # Captura o resumo (entre 'Resumo:' e 'Extrato:')
    summary_match = re.search(
        r"Resumo:\s*(.*?)\s*Extrato:",
        user_input,
        re.IGNORECASE | re.DOTALL
    )
    sic_summary = summary_match.group(1).strip() if summary_match else ""

    # Remove o ponto final do resumo, se houver
    if sic_summary and sic_summary.endswith("."):
        sic_summary = sic_summary[:-1]

    # Captura o extrato (após 'Extrato:')
    extract_match = re.search(
        r"Extrato:\s*(.*)",
        user_input,
        re.IGNORECASE | re.DOTALL
    )
    sic_extract = extract_match.group(1).strip() if extract_match else ""

    return sei_number, sic_summary, sic_extract

def ask_question(question: str) -> str:
    """
    Identifica automaticamente se é um pedido em formato SIC ou não,
    de acordo com a presença de 'SEI ...', 'Pedido ...', 'Resumo:' e/ou 'Extrato:'.
    """
    # Passo 1: buscar documentos relevantes
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "Desculpe, não encontrei informações relevantes nos documentos."

    # Passo 2: concatenar o conteúdo dos documentos em 'context'
    context = "\n".join([doc.page_content for doc in docs])

    # Passo 3: tentar extrair dados de SIC
    sei_number, sic_summary, sic_extract = parse_sic_input(question)

    # Passo 4: determinar o tipo de pedido (SIC ou Geral)
    if sei_number or sic_summary or sic_extract:
        request_type = "SIC"
        # Personaliza o contexto para o pedido SIC
        context = f"Com base nos documentos disponíveis, seguem as informações relevantes para o pedido SEI {sei_number}:\n{context}"
    else:
        request_type = "Geral"

    # Passo 5: montar os argumentos para o LLMChain
    return llm_chain.run(
        context=context,
        question=question,
        request_type=request_type,
        sic_summary=sic_summary,
        sic_extract=sic_extract,
        sei_number=sei_number,
        api_info="",
        additional_resources="",
    )

if __name__ == "__main__":
    print("Chatbot RAG ativo! Digite 'sair' para encerrar.")
    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit"]:
            print("Chatbot encerrado. Até mais!")
            break

        resposta = ask_question(user_input)
        print(f"\nChatbot:\n{resposta}\n")