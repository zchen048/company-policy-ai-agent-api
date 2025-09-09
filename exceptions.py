# ====================
# common exceptions
# ====================

class NoFieldsToUpdateException(Exception):
    """Raised when a PATCH request does not provide any fields to update."""
    pass

# ====================
# user exceptions
# ====================

class UserNotFoundException(Exception):
    """Raised when a user cannot be found in the database."""
    pass

