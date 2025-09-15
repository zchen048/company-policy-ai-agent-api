import re
from ..graph_states import DetailsGraphState
from ..llms import llm
from ...logger import get_logger
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

logger = get_logger(__name__)

def classify_message_intent(state: DetailsGraphState) -> DetailsGraphState:
    '''
    Prompt to classify intent of last user message in relation to effective chat history into 1 of 3 classes
    '''
    logger.debug("-------- Entering intent node --------")

    if state['last_user_message'] == "exit":
        state['last_intent'] = "end"
        logger.debug("-------- User exit of intent node --------")
        return state

    system = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant supporting a company policy query bot. 
        Carefully read the entire chat history and the user's most recent message before making your decision. 
        Your job is to classify the most recent message as one of the following:
        - "Non-policy related" if the message is a greeting, small talk, or not related to company policies.
        - "Policy related — same policy" if the message is a follow-up, clarification, or further question about the same specific company policy discussed earlier in the chat history.
        - "Policy related — different policy" if the message asks about a different company policy than previously discussed.

        Your response must follow the format found between the <result> and </result> tags:
        <result>
        Non-policy related
        </result>
        or
        <result>
        Policy related — same policy
        </result>
        or
        <result>
        Policy related — different policy
        </result>
        Do not provide any further explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    human = '''
        Here is the chat history: {effective_chat_history}
        Here the user's most recent message: {last_user_message}

        How should the most recent message be classified? Please respond only with the result format as shown above.

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm

    try:
        response = chain.invoke(
            {'effective_chat_history': state['effective_chat_history'], 
             'last_user_message': state['last_user_message']
            })
    except Exception as e:
        logger.warning(f"Error in identifying intent: {e}")

    try:
        result_pattern = r'<result>\s*(.*?)\s*</result>'
        result = re.findall(result_pattern, response.content, re.DOTALL)[0]
    except: 
        result = 'None'
        logger.warning("Regex extract intent result None")
    
    state['last_intent'] = result
    logger.info(f"result: {result} added to last intent state")
    
    logger.debug("-------- Normal exit of intent node --------")
    
    return state

def intent_conditional(state: DetailsGraphState) -> str:
    '''
    Determine based on intent whether agent has to ask user again, user user for more information or remove effective context
    '''
    logger.debug("-------- Entering intent conditional edge --------")
    
    if state['last_intent'] == "Policy related — different policy":
        logger.info(f"last intent: {state['last_intent']} -> remove")
        return "remove"
    elif state['last_intent'] == "Policy related — same policy" or state['last_intent'] == "Policy related" :
        logger.info(f"last intent: {state['last_intent']} -> get details")
        return "get details"
    elif state['last_intent'] == "end":
        logger.info(f"last intent: {state['last_intent']} -> exit")
        return "end"
    else:
        logger.info(f"last intent: {state['last_intent']} -> divert")
        return "divert"

def effective_context_removal(state: DetailsGraphState) -> DetailsGraphState:
    '''
    Remove contents in effective chat history and document summary
    '''
    logger.debug("-------- Entering context removal node --------")

    state['effective_chat_history'].clear()
    state['document_summary'] = ''
    
    logger.debug("-------- Normal exit of context removal node --------")
    return state

def get_more_details(state: DetailsGraphState) -> DetailsGraphState:
    '''
    prompt to that allows agent to ask user for details it need for policy query
    '''
    logger.debug("-------- Entering get more details node --------")

    useful_input = state['last_user_message']
    state['effective_chat_history'].append(HumanMessage(content=useful_input))

    system = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant supporting a company policy query bot.
        Your job is to check whether the chat history or the user's query itself contains enough information to accurately retrieve document related to the user's query.
        You must identify if BOTH of the following are present or can be reasonably inferred:
        1. The domain of company policy. IT and HR are examples of domain.
        2. Any context necessary to accurately understand the user's query such as employee role, specific situation, department, or location.
        
        Carefully review the chat history and user's query to determine if all REQUIRED information needed to perform a relevant document search is present. 
        
        You must always reply using the <answer></answer> tags. Do not provide any further explanation.
        If BOTH elements are there, answer "Yes" in the following format:
        <answer>
        Yes
        </answer>
        
        If EITHER element is missing, reply conversationally in <answer></answer> tags to the user, politely asking for the specific information needed. Be clear and avoid asking for obvious details.
        Please answer in the following format:
        <answer>
        Conversationally asking for the information you require.
        </answer>
        Examples of GOOD follow-ups:
        - “Could you let me know which location this applies to?”
        - “Is this for a full-time or contract employee?”
        
        <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    human = '''
        Here is the chat history: {effective_chat_history}

        Please reply in a concise manner using <answer></answer> tags.

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm

    try:
        response = chain.invoke({'effective_chat_history': state['effective_chat_history']})
    except Exception as e:
        logger.warning(f"Error in asking for specific details: {e}")
    
    logger.info(f"response: {response}")

    try:
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer = re.findall(answer_pattern, response.content, re.DOTALL)[0]
        logger.info(f"answer: {answer}")
    except: 
        answer = 'Can you provide me with more details?'
        logger.warning(f"Regex extract error going with default")
    
    if answer != "Yes":
        state['sufficient_details'] = "No"
        state['effective_chat_history'].append(AIMessage(content=answer))
        logger.info(f"Insufficient information provide by user")
    
    logger.debug("-------- Normal exit of get more details node --------")
    return state

def divert_to_policy(state: DetailsGraphState) -> DetailsGraphState:
    '''
    prompt to that allows agent to divert user back to policy questions.
    '''
    logger.debug("-------- Entering divert node --------")

    system = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant supporting a company policy query bot.
        Your task is to politely and concisely guide the user back to asking about company policies from their input.  

        Rules:
        - Do not attempt to answer the user’s non-policy question.  
        - Always redirect them back toward policy in a short, professional, and encouraging way.  
        - Always put your reply in <answer></answer> tags. 
    
        <|eot_id|><|start_header_id|>user<|end_header_id|>
    '''

    human = '''
        Here is the user's input: {last_user_message}

        Please reply in a concise manner using <answer></answer> tags.

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    chain = prompt | llm

    try:
        response = chain.invoke({'last_user_message': state['last_user_message']})
    except Exception as e:
        logger.warning(f'Error in diverting user to policy questions: {e}')
    logger.debug(f"response:{response}")

    try:
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        answer = re.findall(answer_pattern, response.content, re.DOTALL)[0]
    except: 
        answer = 'Do you have any company policies related queries?'
        logger.warning(f"Regex extract error. Default: {answer}")

    state['effective_chat_history'].append(AIMessage(content=answer))
    
    logger.debug("-------- Normal exit of divert node --------")
    return state