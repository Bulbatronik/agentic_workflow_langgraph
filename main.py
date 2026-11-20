import os
import uuid
import subprocess
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


load_dotenv()

llm = init_chat_model('openai:gpt-4.1-mini')

KNOWLEDGE = [
    'Zentrix Labs was founded in 2019 in Tallinn by Mira Okonkwo and Jonas Feld.',
    'Zentrix Labs has 42 employees and its office is on the 7th floor of the Lumen Tower.',
    'The company mascot is a purple otter named Byte.',
    'The flagship product is FlowPilot, a tool that turns spreadsheets into dashboards.',
    'FlowPilot costs 19 EUR per user per month; the Team plan costs 149 EUR per month for up to 10 users.',
    'FlowPilot version 3.2 was released on 4 March 2026 and added dark mode and CSV export.',
    'Support is available Monday to Friday, 9:00-17:00 CET, at support@zentrix.example.',
    'Every Friday at 15:00 the team holds "Otter Hour", a demo session for new features.',
    'Employees get 30 vacation days per year and can work remotely 3 days a week.',
    'The internal Wi-Fi network is called ZTX-Guest and the password changes every Monday.',
]

vector_store = InMemoryVectorStore(OpenAIEmbeddings(model='text-embedding-3-small'))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])


class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user wants to just chat, ask question or change code in the project.')


class State(TypedDict): # Like 'MessagesState'
    messages: Annotated[list, add_messages]
    message_intent: str | None
    next_node: str | None
    coding_request: str | None  # Rewritten instruction for Claude Code (kept out of messages)

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify if the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code").'},
        {'role': 'user', 'content': state['messages'][-1].content} # Last message from a user
    ])

    return {'message_intent': result.message_intent}

def accept_coding(state: State):
    user_prompt = state['coding_request']
    decision = interrupt(f'About to run Claude Code with request:\n\n{user_prompt}\n\nApprove? (yes/no, or type revised request)')

    text = str(decision).strip().lower()

    if text in ['y', 'yes', 'approve', 'ok']:
        return {'next_node': 'coding_agent'}

    if text in ['n', 'no', 'deny', 'cancel']:
        return {'messages':[{'role': 'assistant', 'content': 'Coding request was denied by the user.'}], 'next_node': 'denied'}
    return {'messages': [{'role': 'user', 'content': str(decision).strip()}], 'next_node': 'accept_coding'}
def prompt_llm_chat(state: State):
    print('Using chat')
    
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_rag(state: State):
    print('Using rag')
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=3)

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    messages = [
        {'role': 'system', 'content': f'You are a RAG agent. Answer the user using only the context below. If the answer is not in it, say you don\'t know.\n\nContext:\n{context}'},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_code(state: State):
    print('Using code')
    user_prompt = state['coding_request']
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
    result = subprocess.run(
        ['claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'],
        cwd=workspace,
        capture_output=True,
        text=True
    )  

    output = result.stdout.strip() or result.stderr.strip()  
    
    return {'messages': [{'role': 'assistant', 'content': output}]}


def prepare_coding_request(state: State):
    messages = [
        {'role': 'system', 'content': 'Rewrite the latest user coding request into a clear instruction for Claude Code. Use the conversation history as context. Only output the instruction, no explanation'}
    ] + state['messages']

    response = llm.invoke(messages)
    return {'coding_request': response.content}

def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}


graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('accept_coding', lambda state: 'end' if state.get('next_node') == 'denied' else state['next_node'], {'end': END, 'coding_agent': 'coding_agent', 'accept_coding': 'prepare_coding_request'})
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'prepare_coding_request'})
graph_builder.add_edge('prepare_coding_request', 'accept_coding')

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

config = {'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input('Enter message: ')
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)

    while '__interrupt__' in result: # Handle coding agent (accept/deny)
        prompt = result['__interrupt__'][0].value
        decision = input(f'{prompt}\n> ')
        result = graph.invoke(Command(resume=decision), config=config)
    print(result['messages'][-1].content)