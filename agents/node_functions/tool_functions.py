from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import NUM_OF_DOCS_RETRIEVED, CHROMA_DB_DIR
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection
from logger import get_logger

logger = get_logger(__name__)

@tool
def policy_retrieval_tool(query:str, domain:str):
    '''
    Retrieve company policy documents.

    Parameters:
    - query: The user's input question.
    - domain: The policy domain
    '''

    logger.info(f"Tool call with query:'{query}' and domain:'{domain}' performed.")

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        logger.info(f"Successfully get chromadb client.")
    except Exception as e:
        logger.warning(f"Error in getting chromadb client: {e}.")
        return "Unable to get chroma client"

    try:
        collection_name = "policies"
        collection = client.get_collection(name=collection_name)
        logger.info(f"Get collection '{collection_name}' successfully.")
    except Exception as e:
        logger.warning(f"Unable to get collection '{collection_name}': {e}.")
        return "Unable to find collection"
    
    filter_keys = {"category": domain}

    try:
        query_result = collection.query(
            query_texts=[query],
            n_results=NUM_OF_DOCS_RETRIEVED,
            where=filter_keys
        )
        logger.info(f"Query from collection '{collection_name}' successfully.")
    except Exception as e:
        logger.warning(f"Unable to query collection '{collection_name}': {e}.")
        return "Unable to query collection"

    docs = query_result['documents'][0]
    logger.info(f"{len(docs)} of chunks retrieved.")

    # if not docs retrieved
    if not docs:
        return "No relevant information"

    """
    if domain == "IT":
        collection = "it_policies"
    elif domain == "HR":
        collection = "hr_policies"
    else:
        return "No relevant information"

    # model used by chromadb to embed text chunks in documents
    embeddings = HuggingFaceEmbeddings(model_name="../local_models/arctic-m-sbert")

    # load the appropriate policy collection
    vector_store = Chroma(
        collection_name = collection,
        embedding_function = embeddings,
        persist_directory = "./policies_chroma_db"
    )
    
    # retrieve relevant information based on query
    retriever = vector_store.as_retriever(search_kwargs={"k": NUM_OF_DOCS_RETRIEVED})
    docs = retriever.invoke(query)

    results = []
    for i, doc in enumerate(docs):
        results.append(f'document {i+1}: \n {doc.page_content}')
    """
    
    return "\n".join(docs)

tools = [policy_retrieval_tool]
tools_dict = {tool.name:tool for tool in tools}