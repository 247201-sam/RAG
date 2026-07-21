from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

def index_to_pinecone(chunks, pinecone_api_key, index_name, namespace):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        pinecone_api_key=pinecone_api_key
    )
    return vector_store
