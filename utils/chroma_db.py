from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def add_document(model_path, collection, chromadb_path, pdf_path, chunk_size=500, chunk_overlap=50):
    '''
    Adding a document to a collection chromadb. If collection doesnt exist it will create new one.

    Parameters:
    - model_path (os.path): path to local model
    - collection (str): name of collection in chromadb
    - chromadb_path (os.path): path to chromadb
    - pdf_path (os.path): path to pdf to add to db
    - chunk_size (int): number of char used per embedded vector
    - chunk_overlap (int): num of char overlapped when truncating char for vectors
    '''
    
    # model used by chromadb to embed text chunks in documents
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    # loading the vector store. create one if does not exist
    vector_store = Chroma(
        collection_name = collection,
        embedding_function = embeddings,
        persist_directory = chromadb_path
    )

    # load a specific pdf from file path
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Spliting document up into parts for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # adding document to collection
    vector_store.add_documents(chunks)


if __name__ == "__main__":
    model_path = "../agents/local_models/arctic-embed-m" 
    collection = "hr_policies"
    chromadb_path = "../agents/policy_vector_db"
    pdf_path = "../documents/fake_HR_policies.pdf"
    add_document(model_path, collection, chromadb_path, pdf_path, chunk_size=1000, chunk_overlap=100)
