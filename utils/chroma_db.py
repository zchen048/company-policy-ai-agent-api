import os
from logger import get_logger
from config import EMBED_MODEL_PATH, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from sqlmodel import Field, Session, select
from database import get_session_direct
from exceptions import CollectionNotFoundException, ChunkIDInvalidException, MetadataUpdateException
from models import DocumentDB
from typing import List, Dict, Optional, Any
import hashlib
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection

logger = get_logger(__name__)

# ====================
# chunking utils
# ====================
class ChunkingUtils:
    """
    Utility class for splitting text documents into chunks before storing in ChromaDB.

    This class sets up a recursive character text splitter, which can be
    applied to PDF or plain text content to produce overlapping text chunks
    suitable for embedding and storage.

    Args:
        chunk_size (int): Maximum size of each text chunk before embedding.
        chunk_overlap (int): Number of overlapping characters between text chunks.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int): 
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def path_pdfs_to_document(self, pdf_files: List[str]) -> List[Document]:
        """
        Convert pdfs at specific paths into langchain Documents.

        Args:
            pdf_files (List[str]): A list of file paths to PDF documents.
        
        Returns:
            all_documents (List[Document]): list of pdf files in format of langchain document
        """

        logger.info(f"{len(pdf_files)} document(s) to load")
        all_documents = []

        for file_path in pdf_files:
            try:
                loader = PyPDFLoader(file_path, mode="single")
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded pdf from {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")

        return all_documents

    def dir_pdfs_to_document(self, dir_path: str) -> List[Document]:
        """
        Convert all pdfs in a directory into langchain Documents

        Args:
            dir_path (str): Path to the directory containing PDF files.
        
        Returns:
        all_documents (List[Document]): list of pdf files in format of langchain document

        """
        try:
            # glob for pattern matching, recursive to search subdirectories
            loader = PyPDFDirectoryLoader(dir_path, glob="*.pdf", recursive=True)
            all_documents = loader.load()
            logger.info(f"{len(all_documents)} documents loaded from directory: {dir_path}")
            return all_documents
        except Exception as e:
            logger.error(f"Failed to process PDFs in directory {dir_path}: {e}")
            return []

    def filter_by_hash(
        self,
        all_documents: List[Document], 
    ) -> List[Document]:
        """
        Filter out documents whose content hash already exists in SQLite db.

        Args:
            all_documents (List[Document]): List of Document objects loaded from PDFs.

        Returns:
            filtered (List[Document]): Filtered documents with 'doc_hash' added to metadata.
        """
        filtered = []
        with get_session_direct() as session:
            for doc in all_documents:
                # Hash based on document text
                text = doc.page_content
                hash_val = hashlib.md5(text.encode("utf-8")).hexdigest()

                # Check for existing hash
                statement = select(DocumentDB).where(DocumentDB.hash == hash_val)
                result = session.exec(statement).first()

                if not result:  # new doc → save and keep
                    doc.metadata["doc_hash"] = hash_val # Attach hash into metadata
                    filtered.append(doc)
                    record = DocumentDB(
                        hash=hash_val,
                        source=doc.metadata.get("source", "unknown"),
                    )
                    session.add(record)
                    session.commit()

        return filtered
    
    def split_documents(self, all_documents: List[Document]) -> List[Document]:
        """
        Split pdf documents extracted to chunks.

        Args: 
            all_documents (List[Document]): list of pdf files in format of langchain document

        Returns:
            all_chunks (List[Document]): A list of document chunks extracted from all PDFs.
        """
        all_chunks = self.splitter.split_documents(all_documents)
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def add_metadata_new_chunks(self, chunks:List[Document], keys_to_add:Dict) -> List[Document]:
        """
        Add/modify specific metadata keys to a list of new Document objects.

        Args:
            chunks (List[Document]): List of Document objects to add metadata.
            keys_to_add (Dict): metadata key value pair to be added to each document.

        Returns:
            List[Document]: The same list of Document objects with specified metadata keys added.
        """
        for chunk in chunks:
            for key, value in keys_to_add.items():
                chunk.metadata[key] = value

        return chunks

    def remove_metadata_new_chunks(self, chunks:List[Document], keys_to_remove:List[str]) -> List[Document]:
        """
        Remove specific metadata keys from a list of Document objects.

        Args:
            chunks (List[Document]): List of Document objects to clean.
            keys_to_remove (List[str]): List of metadata keys to remove from each document.

        Returns:
            List[Document]: The same list of Document objects with specified metadata keys removed.
        """
        
        for chunk in chunks:
            for key in keys_to_remove:
                chunk.metadata.pop(key, None)
        
        return chunks

# ====================
# collection utils
# ====================
class CollectionUtils:
    """
    Utility class for managing ChromaDB collections and embeddings.

    This class initializes a persistent ChromaDB client and a
    SentenceTransformer embedding function, which can be used
    for operations like creating collections, inserting chunks,
    updating metadata, and running queries.

    Args:
        chroma_db_dir (str): Path to the directory where ChromaDB will persist data.
        embed_model_path (str): Path or model name for the SentenceTransformer embedding model.
    """
    def __init__(
        self, 
        chroma_db_dir: str,
        embed_model_path: str,
    ):
        self.client = chromadb.PersistentClient(path=chroma_db_dir)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model_path
        )

    def create_collection(self, collection_name:str) -> None:
        """
        Create collection in existing chroma db

        Args:
            collection_name (str): Name given to new collection

        Returns:
            None
        """
        try:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.sentence_transformer_ef
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
    
    def delete_collection(self, collection_name:str) -> None:
        """
        Delete a collection in existing chroma db

        Args:
            collection_name (str): Name of collection to delete
            client (chromadb.PersistentClient): Chroma native client object to perform operations to chroma db

        Returns:
            None
        """
        logger.info(f"Attempting to delete Collection '{collection_name}'. This is not reversible")
        
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")

    def list_collections(self) -> List[Collection]:
        """
        Get list of all collections in an existing chroma db

        Args:
            client (chromadb.PersistentClient): Chroma native client object to perform operations to chroma db

        Returns:
            collections (List[Collection]): A list of Collection objects available in the database.
        """
        try:
            collections = self.client.list_collections()
            logger.info(f"Found {len(collections)} collections.")
            return collections
        except Exception as e:
            logger.info(f"Failed to list collections: {e}")
            return []
    
    def reset_db(self) -> None:
        """
        Empties existing chroma db

        Args:
            client (chromadb.PersistentClient): Chroma native client object to perform operations to chroma db

        Returns:
            None
        """
        logger.info(f"Attempting to reset db. This is not reversible")
        
        try:
            self.client.reset()
            logger.info(f"DB reset successfully.")
        except Exception as e:
            logger.error(f"Failed to delete reset db: {e}")
    
    def get_collection(self, collection_name:str) -> Optional[Collection]:
        """
        Get a specified collection

        Args:
            collection_name (str): Name of collection to get
            client (chromadb.PersistentClient): Chroma native client object to perform operations to chroma db

        Returns:
            collection (Optional[Collection]): The collection object if it exists, otherwise None.
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"Get collection '{collection_name}' successfully.")
            return collection
        except Exception as e:
            logger.warning(f"Unable to get collection '{collection_name}': {e}.")
            return None
      
    def generate_chunk_ids(self, chunks:List[Document]) -> List[Document]:
        """
        Helper function to generate chunk id and place it in chunk's metadata

        Args:
            chunks (List[Document]): pdfs chunks from splitter

        Returns:
            chunks (List[Document]): chunks with their chunk id in their metadata
        """
        hash_count_dic = {}
        for chunk in chunks:
            doc_hash = chunk.metadata.get("doc_hash")
            if not doc_hash:
                raise ValueError("Each chunk must contain 'doc_hash' in metadata before generating IDs.")
            
            hash_count_dic[doc_hash] = hash_count_dic.get(doc_hash, 0) + 1
            chunk_id = f"{doc_hash}_chunk_{hash_count_dic[doc_hash]}"
            chunk.metadata["chunk_id"] = chunk_id

        return chunks
    
    def collection_add_documents(
        self,
        chunks:List[Document],
        collection: Collection
    ) -> None:
        """
        Add chunks from pdfs to specified collection

        Args:
            chunks (List[Document]): pdfs chunks with chunk_id in metadata
            collection (Collection): chroma object that stores chunks

        Returns:
            None
        """
        collection_name = getattr(collection, 'name', 'unknown')

        if collection is None:
            raise CollectionNotFoundException(collection_name)

        try:
            missing_ids = [chunk for chunk in chunks if "chunk_id" not in chunk.metadata]
            if missing_ids:
                raise ChunkIDInvalidException(", ".join(missing_ids))
            
            collection.add(
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                ids=[chunk.metadata["chunk_id"] for chunk in chunks]
            )
            logger.info(f"Add pdfs chunks to collection '{collection_name}' successfully.")
        except Exception as e:
            logger.warning(f"Unable to add chunks to collection '{collection_name}': {e}.")

    def metadata_filter_chunks(
        self,
        collection: Collection,
        filter_keys: Dict
    ) -> Optional[Dict[str, Any]]:
        """
        Filter chunks based on keys in their metadata

        Args:
            collection (Collection): chroma object that stores chunks
            filter_keys (Dict): Keys in for filtering

        Returns:
            filtered_chunks (Dict[str, Any]): Chunks that contains filter_key in their metadata if filtering succeeds.
        """
        collection_name = getattr(collection, 'name', 'unknown')
        
        if collection is None:
            raise CollectionNotFoundException(collection_name)
        
        try:
            filtered_chunks = collection.get(
                where=filter_keys
            )
            logger.info(f"Filter chunks from collection '{collection_name}' successfully.")
            return filtered_chunks
        except Exception as e:
            logger.warning(f"Unable to filter chunks from collection '{collection_name}': {e}.")
            return None

    def delete_chunks(self, collection: Collection, chunks: Dict) -> None:
        """
        Delete chunks from query from a collection

        Args:
            collection (Collection): chroma object that stores chunks
            chunks (Dict[str, Any]): Chunks that are in the result format of a query

        Returns:
            None
        """
        collection_name = getattr(collection, 'name', 'unknown')

        if collection is None:
            raise CollectionNotFoundException(collection_name)
        
        ids_list = chunks.get("ids", [])
        if not ids_list:
            logger.info(f"No chunk IDs found in the query result to delete from collection.")
            return None

        if isinstance(ids_list[0], list):
            ids_list = [item for sublist in ids_list for item in sublist]
        
        try:
            collection.delete(ids=ids_list)
            logger.info(f"Successfully deleted chunks from collection '{collection_name}'.")
        except Exception as e:
            logger.warning(f"Unable to delete chunks from collection '{collection_name}': {e}.")

    def update_chunks_metadata(
        self,
        collection: Collection,
        ids: List,
        keys_to_update: Optional[Dict] = None,
        specfic_metadata: Optional[List[Dict]]= None
    ) -> None:
        """
        Update metadata of chunks given by their id.

        Args:
            collection (Collection): chroma object that stores chunks
            ids (List): list of chunk ids for chunks that will be update
            keys_to_update (Optional[Dict]): metadata key value pair to be added to each document.
            specfic_metadata (Optional[List[Dict]]): Exactly list of metadata to be update for each chunk.

        Returns:
            None
        """
        collection_name = getattr(collection, 'name', 'unknown')

        if keys_to_update and specfic_metadata:
            raise MetadataUpdateException("Provide either `keys_to_update` OR `specific_metadata`, not both.")

        if keys_to_update==None and specfic_metadata==None:
            raise MetadataUpdateException("Provide either `keys_to_update` OR `specific_metadata`")

        if collection is None:
            raise CollectionNotFoundException(collection_name)

        if specfic_metadata and len(ids) != len(specfic_metadata):
            raise ValueError("Mismatch of number of ids and specific metadata.")

        if keys_to_update:
            specfic_metadata = [keys_to_update.copy() for _ in ids]
        
        try:
            collection.update(
                ids=ids,
                metadatas=specfic_metadata
            )
        except Exception as e:
            logger.warning(f"Unable to update chunks in collection:{e}.")

    def query_collection(
        self,
        collection: Collection,
        query_text: str,
        filter_keys: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:

        """
        Query document based on users imput with the option of using metadata for further filtering.

        Args:
            collection (Collection): chroma object that stores chunks
            query_text (str): query from user
            filter_keys (Dict): metadata keys in for filtering

        Returns:
            query_result (Dict[str, Any]): Chunks similar to query text
        """

        collection_name = getattr(collection, 'name', 'unknown')
        
        if collection is None:
            raise CollectionNotFoundException(collection_name)
        
        try:
            query_result = collection.query(
                query_texts=[query_text],
                where=filter_keys
            )
            logger.info(f"Query from collection '{collection_name}' successfully.")
            return query_result
        except Exception as e:
            logger.warning(f"Unable to query collection '{collection_name}': {e}.")
            return None


chunking_helper = ChunkingUtils(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
collection_helper = CollectionUtils(
    chroma_db_dir=CHROMA_DB_DIR,
    embed_model_path=EMBED_MODEL_PATH,
)

# make them available when importing the module
__all__ = ["chunking_helper", "collection_helper", "ChunkingUtils", "CollectionUtils"]


# ====================
# old utils code
# ====================

"""
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

"""