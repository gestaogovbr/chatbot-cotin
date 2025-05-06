from langchain_community.embeddings import DatabricksEmbeddings

def get_embeddings(host, token):
    """Configura e retorna o modelo de embeddings."""
    return DatabricksEmbeddings(
        host=host,
        api_token=token,
        endpoint="databricks-bge-large-en"
    )

def filter_relevant_documents(question, documents, embeddings, keywords, threshold=0.65):
    """Filtra documentos relevantes com base na similaridade e palavras-chave."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    question_embedding = np.array(embeddings.embed_query(question)).reshape(1, -1)
    doc_embeddings = np.array([
        np.array(embeddings.embed_query(doc.page_content))
        for doc in documents
    ])
    similarities = cosine_similarity(question_embedding, doc_embeddings)[0]

    question_lower = question.lower()
    filtered_docs = []
    for doc, sim in zip(documents, similarities):
        doc_lower = doc.page_content.lower()
        keyword_score = sum(1 for kw in keywords if kw in doc_lower)
        final_score = sim + (keyword_score * 0.1)
        if final_score >= threshold:
            filtered_docs.append((doc, final_score))

    filtered_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)
    relevant_docs = [doc for doc, _ in filtered_docs[:5]]
    return relevant_docs if relevant_docs else documents[:5]