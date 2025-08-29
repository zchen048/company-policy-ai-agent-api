import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from .node_functions.tool_functions import tools

load_dotenv() 

# defining the LLM used
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0
)

llm = llm.bind_tools(tools)