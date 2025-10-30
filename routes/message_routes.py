from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
from ..database import get_session
from ..schemas.message_schemas import ReadMessages, LastUserMessage
from ..logic.message_logic import get_chat_eff, get_chat_messages, query_agent

router = APIRouter()

@router.get("/chats/{chat_id}/messages", tags=["Message"], response_model=List[ReadMessages])
def get_chat_messages_endpoint(
    chat_id: int,
    tags=["Message"],
    effective: bool = Query(
        False, description="If true, only return messages used for LLM context"
    ), 
    session: Session = Depends(get_session)):
    """ 
    API endpoint to get all messages in a chat with a particular id 
    - `?effective=true` → only messages used for LLM context  
    - `?effective=false` (default) → all messages
    
    """
    try:
        if not effective:
            messages = get_chat_messages(session=session, chat_id=chat_id)
        else:
            messages = get_chat_eff(session=session, chat_id=chat_id)
    except ChatNotFoundException:
        raise HTTPException(status_code=404, detail="Chat not found")
    return messages

@router.post("/chats/{chat_id}/query")
def query_agent_endpoint(chat_id: int, query: LastUserMessage, session: Session = Depends(get_session)):
    try:
        response = query_agent(session=session, chat_id=chat_id, last_user_message=query.message)
    except ChatNotFoundException:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"Agent response": response}