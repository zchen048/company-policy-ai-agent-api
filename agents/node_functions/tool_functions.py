from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import NUM_OF_DOCS_RETRIEVED

@tool
def policy_retrieval_tool(query:str, domain:str):
    '''
    Retrieve company policy documents.

    Parameters:
    - query: The user's input question.
    - domain: The policy domain either 'HR' or 'IT'.
    '''
    
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

    # if not docs retrieved
    if not docs:
        return "No relevant information"

    results = []
    for i, doc in enumerate(docs):
        results.append(f'document {i+1}: \n {doc.page_content}')
    
    return "\n".join(results)

tools = [policy_retrieval_tool]
tools_dict = {tool.name:tool for tool in tools}