import re
from ..graph_states import GenGraphState
from ..llms import llm
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .tool_functions import tools_dict

def decide_retrieve(state: GenGraphState) -> GenGraphState:
    '''
    Prompt to determine if more company policy information is needed to be retrieved.
    '''
    
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
        print(response)
    except Exception as e:
        response = ToolMessage(content="Error processing input for retrieval")
    
    if response.content == "SKIP":
        response = ToolMessage(content="SKIP")
    
    state['tool_invoke'].append(response)
    
    return state

def need_retrieve(state: GenGraphState) -> GenGraphState:
    '''
    Check if tool call is last message
    '''
    res = state['tool_invoke'][-1]
    return hasattr(res, 'tool_invoke') and len(res.tool_calls)>0

def retrieve_policy(state: GenGraphState) -> GenGraphState:
    '''
    function to retrieve documents retrieved from tool call
    '''

    tool_calls = state['tool_invoke'][-1].tool_calls
    results = []

    for t in tool_calls:
        if not t['name'] in tools_dict:
            result = "Incorrect tool name."
        else:
            query=t['args'].get('query', '')
            domain=t['args'].get('domain', '')
            
            result = tools_dict[t['name']].invoke({
                "query":t['args'].get('query', ''),
                "domain":t['args'].get('domain', '')
            })
            print(result)
        results.append(
            ToolMessage(
                tool_call_id = t['id'], 
                name=t['name'], 
                content=str(result)
            ))
    
    state['tool_invoke'].append(results)
    return state

def document_summary(state: GenGraphState) -> GenGraphState:
    '''
    Prompt to summarise documents retrieved in relation to user query.
    '''
    if type(state['tool_invoke'][-1][-1]) != ToolMessage:
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
        print(f'Error in Summarizing Information: {e}')

    try:
        summary_pattern = r'Summary:\s*(.*?)(?=\n|\Z)'
        summary = re.findall(summary_pattern, response.content, re.DOTALL)[0]
    except: 
        summary = 'None'
    
    state['document_summary'] = summary

    return state

def check_context_length(state: GenGraphState) -> GenGraphState:
    '''
    Function that uses the LLM tokeniser to check if prompt will be greater than context window
    '''
    state['within_token_limit'] = "Yes"
    return state

def context_length_conditional(state: GenGraphState) -> str:
    '''
    Condtional if trucation of effective chat history is needed
    '''
    if state['within_token_limit'] == "Yes":
        return "generate"
    else:
        return "truncate"

def truncate_chat_history(state: GenGraphState) -> GenGraphState:
    '''
    Truncates the chat history to ensure prompt does not exceed context window
    '''
    return state

def answer_user_query(state: GenGraphState) -> GenGraphState:
    '''
    Prompt that uses retrieved context and chat history to answer user's input query.
    '''
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
        print(response)
    except Exception as e:
        response = "Error processing input"
    
    state['effective_chat_history'].append(response)

    return state
