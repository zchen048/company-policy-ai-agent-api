from typing import Optional, List, Union
from sqlmodel import Session, select
from ..models import RoleEnum, Chat, Message
from ..utils import sg_datetime
from ..exceptions import ChatNotFoundException
from ..logger import get_logger

logger = get_logger(__name__)

def get_chat_eff(session: Session, chat_id: int) -> List[Message]:
    """ Get effective LLM context message of a chat."""
    # Check if chat exist
    chat = session.get(Chat, chat_id)
    if not chat:
        raise ChatNotFoundException()
    
    # query DB for effective messages with chat id
    messages = session.exec(
        select(Message).where(Message.chat_id == chat_id, Message.effective == True)
    ).all()
    return chats

def get_chat_messages(session: Session, chat_id: int) -> List[Message]:
    """ Get all message of a chat."""
    # Check if chat exist
    chat = session.get(Chat, chat_id)
    if not chat:
        raise ChatNotFoundException()
    
    # query DB for all message with chat id
    messages = session.exec(
        select(Message).where(Message.chat_id == chat_id)
    ).all()
    return chats