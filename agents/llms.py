import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from .node_functions.tool_functions import tools

load_dotenv() 

# defining the LLM used
llm = ChatGroq(
    model="llama-3.1-8b-instant", # llama3-8b-8192
    temperature=0
)

llm2 = ChatGroq(
    model="llama-3.1-8b-instant", # llama3-8b-8192
    temperature=0
)

llm2 = llm2.bind_tools(tools)
