from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from ..database import get_session
from ..schemas.chat_schemas import ReadChat, ReadState, UpdateChat
from ..logic.chat_logic import create_chat, get_user_chats, get_chat_by_id

router = APIRouter()

@router.post("/users/{user_id}/chats", response_model=ReadChat)
def create_chat_endpoint(user_id: int, session: Session = Depends(get_session)):
    try:
        chat = create_chat(session=session, user_id=user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")
    return chat

@router.get("/users/{user_id}}/chats", response_model=List[ReadChat])
def get_user_chats_endpoint(user_id: int, session: Session = Depends(get_session)):
    try:
        chats = get_user_chats(session=session, user_id=user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")
    return chats

@router.get("/chats/{chat_id}", response_model=ReadChat)
def get_chat_by_id_endpoint(chat_id: int, session: Session = Depends(get_session)):
    chat = get_chat_by_id(session=session, id=chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.get("/chats/{chat_id}/state", response_model=ReadState)
def get_state_by_id_endpoint(chat_id: int, session: Session = Depends(get_session)):
    chat = get_chat_by_id(session=session, id=chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.patch("/chats/{chat_id}", response_model=ReadChat)
def update_chat_endpoint(chat_id: int, to_change: UpdateChat, session: Session = Depends(get_session)):
    try:
        chat = update_chat(session=session, id=chat_id, data=to_change)
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat_endpoint(chat_id: int, session: Session = Depends(get_session)):
    deleted = delete_chat(session=session, id=chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")