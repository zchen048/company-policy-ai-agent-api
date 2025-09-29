from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from .utils import sg_datetime

class RankEnum(str, Enum):
    EXECUTIVE = "Executive"
    SENIOR_EXECUTIVE = "Senior executive"
    ASSISTANT_MANAGER = "Assistant manager"
    MANAGER = "Manager"
    VICE_PRESIDENT = "Vice president"

class IntentEnum(str, Enum):
    NON_POLICY_RELATED = "Non-policy related"
    POLICY_RELATED = "Policy related"
    SAME_POLICY = "Policy related — same policy"
    DIFFERENT_POLICY = "Policy related — different policy"
    END = "end"

class RoleEnum(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class User(SQLModel, table=True):
    """
    Represents a user object. 
    
    Attributes:
        id: Unique identifier of user. 
        name: Full name of user.
        email: Email give to user.
        department: Department name.
        rank: User rank within the company.
        title: Job title of the user.
        created_at: Timestamp when the user was created.
        modified_at: Timestamp when the user was last modified.
    """

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(index=True, unique=True)
    department: str
    rank: RankEnum
    title: str
    created_at: datetime = Field(default_factory=sg_datetime.get_sgt_time)
    modified_at: datetime = Field(default_factory=sg_datetime.get_sgt_time)

    chats: List["Chat"] = Relationship(back_populates="user")

class Chat(SQLModel, table=True):
    """
    Represents a chat object. 
    
    Attributes:
        id: Unique identifier of chat.
        user_id: User id of owner of this chat
        title: chat title currently function based generation user-chat-number
        created_at: Timestamp when the chat was created.
        modified_at: Timestamp when the chat was last modified.
        last_intent: intent of last user enter input
        document_summary: Summary of RAG document retrieved related to user query
        sufficient_details: Boolean to decide subgraph
        within_token_limit: Boolean to decide if chat history need to be truncated
    """

    __tablename__ = "chats"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True)
    title: str = Field(default="Untitled Chat")
    created_at: datetime = Field(default_factory=sg_datetime.get_sgt_time)
    modified_at: datetime = Field(default_factory=sg_datetime.get_sgt_time)
    
    # Agent state variables
    last_intent: Optional[IntentEnum] = Field(default=None)
    document_summary: Optional[str] = Field(default=None)
    sufficient_details: bool = Field(default=False)
    within_token_limit: bool = Field(default=True)

    user: User = Relationship(back_populates="chats")
    messages: List["Message"] = Relationship(back_populates="chat")

class Message(SQLModel, table=True):
    """
    Represents a message object. 
    
    Attributes:
        id: Unique identifier of message.
        chat_id: Chat id of the chat that message belongs to
        role: user | assistant 
        content: Message body content
        effective: boolean to decide if message will be used in generation of answer
        created_at: Timestamp when the message was created.
 
    """
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: int = Field(foreign_key="chats.id")
    role: RoleEnum
    content: str
    effective: bool = Field(default=True)  
    created_at: datetime = Field(default_factory=sg_datetime.get_sgt_time)
    
    chat: Chat = Relationship(back_populates="messages")


class Document(SQLModel, table=True):
    """
    Represents a document object. 
    
    Attributes:
        hash: Unique identifier of message.
        source: Chat id of the chat that message belongs to
        created_at: Timestamp when the message was created.
 
    """
    hash: str = Field(primary_key=True)
    source: str
    created_at: datetime.datetime = Field(default_factory=sg_datetime.get_sgt_time)