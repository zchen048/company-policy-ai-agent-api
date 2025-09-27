import re
from logger import get_logger
from ..graph_states import GenGraphState
from ..llms import llm
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .tool_functions import tools_dict

logger = get_logger(__name__)

def decide_retrieve(state: GenGraphState) -> GenGraphState:
    '''
    Prompt to determine if more company policy information is needed to be retrieved.
    '''

    logger.debug("-------- Entering decide retrieve node --------")
    
    system = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant with access to company policy documents in two domains: HR and IT.
        Your task is to decide whether you need to retrieve additional information from the policy database to answer the user's question.

        - If you already know the answer or the question is general knowledge, respond with: "SKIP".
        - If you need more context from the policy documents, perform a tool call for policy_retrieval_tool.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''
    human = '''
        Here is the user input:
        {last_human_message}
    '''

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm 

    try:
        response = chain.invoke({'last_human_message': state['last_human_message']})
        logger.info(f"{response} was returned by llm in decide retrieve node")
    except Exception as e:
        logger.warning(f"Error processing input for retrieval")
        response = ToolMessage(content="Error processing input for retrieval")
    
    if response.content == "SKIP":
        logger.info(f"Skip retrieval")
        response = ToolMessage(content="SKIP")
    
    state['tool_invoke'].append(response)
    logger.info(f"response: {response} added to tool_invoke state")

    logger.debug("-------- Normal exit of decide retrieve node --------")
    return state

def need_retrieve(state: GenGraphState) -> GenGraphState:
    '''
    Check if tool call is last message
    '''
    logger.debug("-------- Entering retrieve check conditional edge --------")
    
    res = state['tool_invoke'][-1]
    return hasattr(res, 'tool_invoke') and len(res.tool_calls)>0

def retrieve_policy(state: GenGraphState) -> GenGraphState:
    '''
    function to retrieve documents retrieved from tool call
    '''

    logger.debug("-------- Entering retrieve policy node --------")

    tool_calls = state['tool_invoke'][-1].tool_calls
    results = []

    for t in tool_calls:
        if not t['name'] in tools_dict:
            logger.warning(f"tool with incorrect name was called")
            result = "Incorrect tool name."
        else:
            query=t['args'].get('query', '')
            domain=t['args'].get('domain', '')
            
            result = tools_dict[t['name']].invoke({
                "query":t['args'].get('query', ''),
                "domain":t['args'].get('domain', '')
            })
            logger.info(f"result: {result} from a tool call performed")
        results.append(
            ToolMessage(
                tool_call_id = t['id'], 
                name=t['name'], 
                content=str(result)
            ))
    
    state['tool_invoke'].append(results)
    logger.info(f"{results} have been added to from tool_invoke state")
    logger.debug("-------- Normal exit of retrieve policy node --------")
    return state

def document_summary(state: GenGraphState) -> GenGraphState:
    '''
    Prompt to summarise documents retrieved in relation to user query.
    '''

    logger.debug("-------- Entering document summary node --------")

    if type(state['tool_invoke'][-1][-1]) != ToolMessage:
        logger.info("No document match query, retrival result is empty.")
        return state
    
    documents = state['tool_invoke'][-1][-1].content
    query = state['last_human_message']

    system = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant whose task is to summarize information from multiple documents.
        Given a set of documents retrieved in response to a user query, carefully read and analyze the content.
        Your job is to write a concise and informative summary that directly addresses the user query using only information found in the provided documents.
        - Focus only on information that is relevant to the user's query.
        - If there is any disagreement or conflict between the documents about a fact or detail, do not include that information in your summary.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    human = '''
        Here is the query: {query}
        Here are the documents: {documents}

        Summarize the contents of the documents into a single paragraph of less than 50 words.
        If none of the documents are relevant, return one word - 'None'.
        Do not return any explanation.

        Your response must follow the format found between the <answer> and </answer> tags.
        <answer>
        Summary: <insert summary if document is relevant>
        </answer>

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm

    try:
        response = chain.invoke({'query': query, 'documents': documents})
        print(response)
    except Exception as e:
        logger.warning(f"Error in Summarizing Information: {e}")

    try:
        summary_pattern = r'Summary:\s*(.*?)(?=\n|\Z)'
        summary = re.findall(summary_pattern, response.content, re.DOTALL)[0]
    except: 
        logger.warning(f"Regex for Summarizing node output failed, default summary is none: {e}")
        summary = 'None'
    
    state['document_summary'] = summary
    logger.info(f"Summary:{summary} was added document_summary state ")
    
    logger.debug("-------- Normal exit of document summary node --------")
    return state

def check_context_length(state: GenGraphState) -> GenGraphState:
    '''
    Function that uses the LLM tokeniser to check if prompt will be greater than context window
    '''
    logger.debug("-------- Entering check context length node --------")
    # Can't permission for llama huggingface to get tokeniser 
    state['within_token_limit'] = "Yes"
    logger.debug("-------- Normal exit of check context length node --------")
    return state

def context_length_conditional(state: GenGraphState) -> str:
    '''
    Condtional if trucation of effective chat history is needed
    '''

    logger.debug("-------- Entering context length check conditional edge --------")

    if state['within_token_limit'] == "Yes":
        logger.info("Context length is within limit")
        return "generate"
    else:
        logger.info("Context length exceeds limit")
        return "truncate"

def truncate_chat_history(state: GenGraphState) -> GenGraphState:
    '''
    Truncates the chat history to ensure prompt does not exceed context window
    '''
    logger.debug("-------- Entering truncate chat history node --------")
    state['effective_chat_history'] = state['effective_chat_history'][4:]
    logger.info("Remove 2 oldest user-llm interaction ie 4 messages")
    logger.debug("-------- Normal exit of truncate chat history node --------")
    return state

def answer_user_query(state: GenGraphState) -> GenGraphState:
    '''
    Prompt that uses retrieved context and chat history to answer user's input query.
    '''

    logger.debug("-------- Entering answer user query node --------")

    system = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a highly knowledgeable, concise, and unbiased AI assistant.
        Your job is to answer user questions as accurately and helpfully as possible, using all available information.
        
        - You have access to two additional sources of information:
            1. Externally retrieved information provided in <context></context>.
            2. Chat history provided in <messages></messages>.
        - Always use relevant information from context and messages to generate your response to user input.
        - Provide the most direct and relevant answer first, followed by a brief explanation if necessary.
        - If you do not know the answer, state this honestly.
        - Maintain a neutral, professional, and helpful tone. 
        
        <context>
        {context}
        </context>

        <messages>
        {messages}
        <messages/>

        <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''
    
    human = '''
        Here is the user input:
        {last_human_message}

        Carefully analyze the input and provide concise answer.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm 

    try:
        response = chain.invoke({
            'last_human_message': state['last_human_message'],
            'messages': state['effective_chat_history'],
            'context': state['document_summary']
        })
        logger.info(f"generated response from llm: {response} ")
    except Exception as e:
        logger.warning(f"Error occured during generation: {e}")
        response = "Error generating"
    
    state['effective_chat_history'].append(response)
    
    logger.debug("-------- Normal exit of answer user query node --------")
    return state
