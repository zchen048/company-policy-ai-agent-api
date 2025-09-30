from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from .. models import IntentEnum

class ReadChat(BaseModel):
    id: int = Field(..., description="id given to the chat")
    user_id: int = Field(..., description="id of the user who owns the chat")
    title: str = Field(..., description="Title of the chat")
    created_at: datetime = Field(..., description="Datetime of chat creation")
    modified_at: datetime = Field(..., description="Datetime for when chat was last modified")
    last_intent: Optional[IntentEnum] = Field(..., description="Intent class of last message")
    document_summary: Optional[str] = Field(..., description="summary of RAG documents retrieved")
    sufficient_details: bool = Field(..., description="Bool to decide subgraph invoked")
    within_token_limit: bool = Field(..., description="Bool to check if prompt is within token limit")

    class Config:
        orm_mode = True

class ReadState(BaseModel):
    last_intent: Optional[IntentEnum] = Field(..., description="Intent class of last message")
    document_summary: Optional[str] = Field(..., description="summary of RAG documents retrieved")
    sufficient_details: bool = Field(..., description="Bool to decide subgraph invoked")
    within_token_limit: bool = Field(..., description="Bool to check if prompt is within token limit")

    class Config:
        orm_mode = True

class UpdateChat(BaseModel):
    title: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="New title for the chat"
    )