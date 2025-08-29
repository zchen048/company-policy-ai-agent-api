from typing import TypedDict, List
from langchain_core.messages import BaseMessage, ToolMessage

class DetailsGraphState(TypedDict):
    last_user_message:str
    effective_chat_history:List[BaseMessage]
    last_intent:str
    sufficient_details:str
    document_summary:str

class GenGraphState(TypedDict):
    last_user_message:str
    effective_chat_history:List[BaseMessage]
    document_summary:str
    within_token_limit:str
    tool_invoke:List[ToolMessage]