from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL_NAME

STRICT_PROMPT = """
You are a strict QA assistant. Answer the user's question strictly using ONLY the context provided below. 

Strict Rules:
1. If the answer is not contained in the context, output EXACTLY: "The answer is not available in the provided document."
2. Do NOT extrapolate or use outside knowledge.

Context:
{context}

Question: {question}
Answer:
"""

def generate_answer(query, vector_store, groq_api_key, top_k, similarity_threshold, namespace):
    docs_and_scores = vector_store.similarity_search_with_score(
        query, k=top_k, namespace=namespace
    )

    filtered_docs = [
        (doc, score) for doc, score in docs_and_scores if score >= similarity_threshold
    ]

    if not filtered_docs:
        return "The answer is not available in the provided document.", []

    context_text = "\n\n".join([doc.page_content for doc, _ in filtered_docs])
    prompt = PromptTemplate(template=STRICT_PROMPT, input_variables=["context", "question"])
    
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=LLM_MODEL_NAME,
        temperature=0
    )
    
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": query})
    
    sources = [
        {
            "page": doc.metadata.get("page", "N/A"),
            "text": doc.page_content[:300] + "...",
            "score": score
        } for doc, score in filtered_docs
    ]

    return response.content, sources
