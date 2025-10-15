from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from ..database import get_session
from ..schemas.chat_schemas import ReadChat, ReadState, UpdateChat
from ..logic.chat_logic import create_chat, get_user_chats, get_chat_by_id, update_chat, delete_chat
from ..exceptions import UserNotFoundException, ChatNotFoundException, NoFieldsToUpdateException

router = APIRouter()

@router.post("/users/{user_id}/chats", tags=["Chat"], response_model=ReadChat)
def create_chat_endpoint(user_id: int, session: Session = Depends(get_session)):
    """ API endpoint to create a new chat for a user. """
    try:
        chat = create_chat(session=session, user_id=user_id)
    except UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found")
    return chat

@router.get("/users/{user_id}/chats", tags=["Chat"], response_model=List[ReadChat])
def get_user_chats_endpoint(user_id: int, session: Session = Depends(get_session)):
    """ API endpoint to get all chats of a user. """
    try:
        chats = get_user_chats(session=session, user_id=user_id)
    except UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found")
    return chats

@router.get("/chats/{chat_id}", tags=["Chat"], response_model=ReadChat)
def get_chat_by_id_endpoint(chat_id: int, session: Session = Depends(get_session)):
    """ API endpoint to get a chat of a particular id. """
    try:
        return get_chat_by_id(session=session, id=chat_id)
    except ChatNotFoundException:
        raise HTTPException(status_code=404, detail="Chat not found")

@router.get("/chats/{chat_id}/state", tags=["Chat"], response_model=ReadState)
def get_state_by_id_endpoint(chat_id: int, session: Session = Depends(get_session)):
    """ API endpoint to get a chat fields used by langGraph as state. """
    try:
        return get_chat_by_id(session=session, id=chat_id)
    except ChatNotFoundException:
        raise HTTPException(status_code=404, detail="Chat not found")

@router.patch("/chats/{chat_id}", tags=["Chat"], response_model=ReadChat)
def update_chat_endpoint(chat_id: int, to_change: UpdateChat, session: Session = Depends(get_session)):
    """ API endpoint update info of chat of a particular id. """
    try:
        chat = update_chat(session=session, id=chat_id, data=to_change)
    except ChatNotFoundException:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NoFieldsToUpdateException:
        raise HTTPException(status_code=400, detail="No fields provided to update")
    return chat

@router.delete("/chats/{chat_id}", tags=["Chat"], status_code=204)
def delete_chat_endpoint(chat_id: int, session: Session = Depends(get_session)):
    """ API endpoint to delete chat of a particular id. """
    try:
        return delete_chat(session=session, id=chat_id)
    except ChatNotFoundException:
        raise HTTPException(status_code=404, detail="Chat not found")