from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from models import RoleEnum

class ReadMessages(BaseModel):
    id: int = Field(..., description="id given to the message")
    chat_id: int = Field(..., description="id of the chat that message is in")
    role: RoleEnum = Field(..., description="user | assistant ")
    content :str = Field(..., description="Message text content")
    created_at: datetime = Field(..., description="Datetime of chat creation")

    class Config:
        orm_mode = True

class LastUserMessage(BaseModel):
    message: str

