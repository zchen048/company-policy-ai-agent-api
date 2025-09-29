from typing import Optional, List, Union
from sqlmodel import Session, select, func
from ..models import IntentEnum, Chat
from ..utils import sg_datetime
from ..schemas.chat_schemas import UpdateChat
from ..exceptions import UserNotFoundException, ChatNotFoundException
from ..logger import get_logger

logger = get_logger(__name__)

def create_chat(session: Session, user_id: int) -> Chat:
    """ Create and persist a new chat for a user in the database."""
    # check if user exist
    user = session.get(User, user_id)
    if not user:
        raise UserNotFoundException()

    # get the front name of user and used for chat title
    first_name = user.name.split(" ")[0] 
    
    # count the number of chats user has
    existing_count = session.exec(
        select(func.count()).where(Chat.user_id == user_id)
    ).one()
    
    # Just give the title 
    title = f"{first_name}-chat-{existing_count + 1}"

    chat = Chat(
        user_id=user_id, 
        title=title
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)  # refresh to get generated ID
    logger.info(f"Chat `{title}` created for {user.name} ")
    return chat

def get_user_chats(session: Session, user_id: int) -> List[Chat]:
    """ Get all chats of a user."""
    # Check if user exist
    user = session.get(User, user_id)
    if not user:
        raise UserNotFoundException()
    
    # query DB for all chats with user id
    chats = session.exec(
        select(Chat).where(Chat.user_id == user_id)
    ).all()
    return chats

def get_chat_by_id(session: Session, id: int) -> Chat:
    """ Get chat with a particular id. """
    chat = session.get(Chat, id)
    if not chat:
        raise ChatNotFoundException()
    return chat

def update_chat(session: Session, id: int, data: UpdateChat) -> Chat:
    """ Update specified fields of chat given by specific id. """
    chat = session.get(Chat, id)
    if not chat:
        raise ChatNotFoundException()
    
    updates = data.model_dump(exclude_unset=True)
    if not updates:
        raise NoFieldsToUpdateException()

    for field, value in updates.items():
        setattr(chat, field, value)
    
    chat.modified_at = sg_datetime.get_sgt_time()

    session.add(chat)
    session.commit()
    session.refresh(chat)
    logger.info(f"Chat `{chat.id}` updated. ")
    return chat
    
def delete_chat(session: Session, id: int):
    """ delete chat given by specific id. """
    chat = session.get(Chat, id)
    if not chat:
        raise ChatNotFoundException()
    
    session.delete(chat)
    session.commit()
    logger.info(f"Chat `{chat.id}` deleted. ")

