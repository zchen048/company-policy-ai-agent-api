from typing import Optional, List, Union
from sqlmodel import Session, select
from sqlalchemy import update
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

    logger.info(f"Getting all effective chat messages from chat of id `{chat_id}`")
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
    logger.info(f"Getting all chat messages from chat of id `{chat_id}`")
    return chats

def query_agent(session: Session, chat_id: int, last_user_message: str) -> str:
    # Get necessary state: chat_history, document_summary, last_intent
    logger.debug(f"Getting necessary state: chat_history, document_summary, last_intent")
    chat_statement = (
        select(Message)
        .where(Message.chat_id == chat_id, Message.effective == True)
        .order_by(Message.created_at.asc())  # oldest first
    )
    messages = session.scalars(chat_statement).all()

    others_statement = (
        select(
            Chat.document_summary,
            Chat.last_intent
        )
        .where(Chat.id == chat_id)
    )

    document_summary = ""
    last_intent = ""
    result = session.execute(others_statement).first()

    effective_chat_history = [{m.role: m.content} for m in messages]

    if result:
        document_summary, last_intent = result

    if not document_summary:
        document_summary = ""

    if last_intent=='end':
        return "Chat has ended"
    
    # Invoking of details graph
    logger.debug(f"Invoking of details graph")
    details_graph_state = {
        'last_user_message': last_user_message,
        'effective_chat_history': effective_chat_history,
        'last_intent': "",
        'sufficient_details': "",
        'document_summary':document_summary
    }
    
    details_graph_state = details_graph.invoke(details_graph_state)

    # context removal occurs
    if effective_chat_history and len(details_graph_state['effective_chat_history']) <= 2:
        ineffective_stmt = (
            update(Message)
            .where(Message.chat_id == chat_id, Message.effective == True)
            .values(effective=False)
        )
        session.execute(ineffective_stmt)
        session.commit()
    
    # Invoking of gen graph
    logger.debug(f"Invoking of gen graph")
    
    # Sufficient_details for RAG
    if details_graph_state['sufficient_details'] == "Yes":
        gen_graph_state = {
            'last_user_message': last_user_message,
            'effective_chat_history': details_graph_state['effective_chat_history'], # in case of context removal
            'document_summary': details_graph_state['document_summary'], # in case of context removal
            'within_token_limit':"",
            'tool_invoke':[]
        }

        gen_graph_state = gen_graph.invoke(gen_graph_state)
        new_messages = gen_graph_state['effective_chat_history'][-2:]
        document_summary_to_save = gen_graph_state['document_summary']
    
    else:
        new_messages = details_graph_state['effective_chat_history'][-2:]
        document_summary_to_save = details_graph_state['document_summary']
    
    logger.info(f"new messages {new_messages} to be added to db")
    
    for msg in new_messages:
        if isinstance(msg, HumanMessage):
            role = RoleEnum.USER
        elif isinstance(msg, AIMessage):
            role = RoleEnum.ASSISTANT
        else:
            logger.warning(f"Message {msg} with unsupported type: {type(msg)}")
            continue 
            
        new_message = Message(
            chat_id=chat_id,
            role=role,
            content=msg.content,
            effective=True,
        )
        session.add(new_message)
    session.commit()

    stmt = (
        update(Chat)
        .where(Chat.id == chat_id)
        .values(document_summary=document_summary_to_save, last_intent=details_graph_state['last_intent'])
    )
    session.execute(stmt)
    session.commit()

    return new_messages[-1].content if new_messages else "No response generated."