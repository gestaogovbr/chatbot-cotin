import chainlit as cl

async def start_chat(memory):
    """Inicializa o chat e envia mensagem de boas-vindas."""
    memory.clear()
    await cl.Message(
        content="Olá! Sou o Cotin IA, pronto para ajudar com dados abertos de compras públicas.\nDigite sua pergunta!"
    ).send()

async def handle_message(msg, ask_question_func, retriever, llm_chain, embeddings, keywords, threshold):
    """Processa a mensagem do usuário e retorna a resposta com fontes e feedback."""
    user_text = msg.content if hasattr(msg, "content") else str(msg)
    if not isinstance(user_text, str):
        user_text = str(user_text)

    # 1) Cria uma mensagem "placeholder" para sinalizar que o bot está pensando.
    placeholder = await cl.Message(content="Processando, por favor aguarde...").send()

    # 2) Gera a resposta pelo LLM e obtém documentos fonte
    resposta, source_documents = ask_question_func(
        user_text,
        retriever,
        llm_chain,
        embeddings,
        keywords,
        threshold,
        return_sources=True
    )

    # 3) Prepara as fontes para exibição
    sources = []
    if source_documents:
        for i, doc in enumerate(source_documents[:3]):  # Limita a 3 fontes para não sobrecarregar a UI
            source = f"{doc.metadata.get('source', 'Desconhecido')}"
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            sources.append(cl.Text(content=content_preview, name=f"Fonte {i+1}: {source}"))

    # 4) Adiciona botões de feedback - CORRIGIDO para incluir payload
    actions = [
        cl.Action(
            name="útil",
            value="útil",
            description="Esta resposta foi útil",
            payload={"question": user_text, "answer": resposta}  # Usando payload em vez de context
        ),
        cl.Action(
            name="não_útil",
            value="não_útil",
            description="Esta resposta não foi útil",
            payload={"question": user_text, "answer": resposta}  # Usando payload em vez de context
        )
    ]

    # 5) Atualiza a mensagem de placeholder com o conteúdo final, fontes e ações
    placeholder.content = resposta
    placeholder.elements = sources
    placeholder.actions = actions
    await placeholder.update()

# Manipulador de feedback do usuário
@cl.action_callback("útil")
async def on_useful_feedback(action):
    # Registra feedback positivo
    await cl.Message(content="Obrigado pelo feedback positivo! 👍").send()
    # Salva o feedback
    try:
        save_feedback(
            question=action.payload.get("question"),  # Usando payload em vez de context
            answer=action.payload.get("answer"),      # Usando payload em vez de context
            feedback="positive",
            user_id=cl.user_session.get("user_id", "anônimo")
        )
    except Exception as e:
        print(f"Erro ao salvar feedback: {e}")

@cl.action_callback("não_útil")
async def on_not_useful_feedback(action):
    # Registra feedback negativo e pede mais informações
    await cl.Message(content="Lamento que a resposta não tenha sido útil. Poderia detalhar o que faltou ou como poderia melhorar?").send()
    # Salva o feedback
    try:
        save_feedback(
            question=action.payload.get("question"),  # Usando payload em vez de context
            answer=action.payload.get("answer"),      # Usando payload em vez de context
            feedback="negative",
            user_id=cl.user_session.get("user_id", "anônimo")
        )
    except Exception as e:
        print(f"Erro ao salvar feedback: {e}")

# Função simples para salvar feedback
def save_feedback(question, answer, feedback, user_id=None):
    """Salva o feedback do usuário para análise posterior."""
    import json
    import os
    from datetime import datetime

    feedback_dir = "feedback"
    os.makedirs(feedback_dir, exist_ok=True)

    feedback_data = {
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat()
    }

    # Gera um nome de arquivo único
    filename = f"{feedback_dir}/feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=2)

    return filename