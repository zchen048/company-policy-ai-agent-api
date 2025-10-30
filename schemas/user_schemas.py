from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from models import RankEnum

class CreateUser(BaseModel):
    name: str = Field(
        ..., 
        min_length=2, 
        max_length=50, 
        description="Full name of the user",
        example="Benson Tan"
    )
    email: str = Field(
        ..., 
        pattern=r'^\S+@\S+\.\S+$',
        description="Valid email address of user",
        example="benson@company.com"
    )
    department: str = Field(
        ..., 
        min_length=2, 
        max_length=50, 
        description="Department user is from",
        example="Human resource"
    )
    rank: RankEnum = Field(
        ..., 
        description="Rank of user. Executive | Senior executive | Assistant manager | Manager | Vice president",
        example="Executive"
    )
    title: str = Field(
        ..., 
        min_length=2, 
        max_length=50, 
        description="User's job title",
        example="HR executive"
    )


class ReadUser(CreateUser):
    id: int = Field(..., description="id given to the user")
    created_at: datetime = Field(..., description="Datetime of user creation")
    modified_at: datetime = Field(..., description="Datetime for when user was last modified")

    class Config:
        orm_mode = True

class UpdateUser(BaseModel):
    name: Optional[str] = Field(
        None, 
        min_length=2, 
        max_length=50, 
        description="Updated name of user",
        example="Benson Tan"
    )
    email: Optional[str] = Field(
        None, 
        pattern=r'^\S+@\S+\.\S+$',
        description="Updated email address of user",
        example="benson@company.com"
    )
    department: Optional[str] = Field(
        None, 
        min_length=2, 
        max_length=50, 
        description="Updated department of user",
        example="Finance"
    )
    rank: Optional[RankEnum] = Field(
        None, 
        description="Updated rank of user. Executive | Senior executive | Assistant manager | Manager | Vice president",
        example="Executive"
    )
    title: Optional[str] = Field(
        None, 
        min_length=2, 
        max_length=50, 
        description="User's updated job title",
        example="Finance executive"
    )
