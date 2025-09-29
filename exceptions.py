# ====================
# common exceptions
# ====================

class NoFieldsToUpdateException(Exception):
    """Raised when a PATCH request does not provide any fields to update."""
    pass

# ====================
# chroma utils exceptions
# ====================

class CollectionNotFoundException(Exception):
    """Raised when a requested collection name is not found in Chroma db"""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        super().__init__(f"Collection '{collection_name}' not found in Chroma DB.")

class ChunkIDInvalidException(Exception):
    """Raised when chunk with id specified is not found"""
    def __init(self, chunk_id: str):
        self.chunk_id = chunk_id
        super().__init__(f"Chunk id '{chunk_id}' not found in collection.")

class MetadataUpdateException(Exception):
    """Raised when updating metadata in Chroma fails"""
    pass

# ====================
# user exceptions
# ====================

class UserNotFoundException(Exception):
    """Raised when a user cannot be found in the database."""
    pass

# ====================
# chat exceptions
# ====================

class ChatNotFoundException(Exception):
    """Raised when a chat cannot be found in the database."""
    pass

