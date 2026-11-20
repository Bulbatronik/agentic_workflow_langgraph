import os
import uuid
import time
import subprocess
import warnings
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from ddgs import DDGS # For the web search

from langchain.chat_models import init_chat_model
warnings.filterwarnings('ignore', message='.*langchain-community', category=DeprecationWarning)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


load_dotenv()

llm = init_chat_model('openai:gpt-4.1-mini')

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs')
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faiss_index')  # Delete this folder to re-index docs/
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')  # Where Claude Code works

os.makedirs(WORKSPACE_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

if os.path.exists(INDEX_DIR):  # Load the saved index (no embedding calls)
    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)  # Safe: we created this file ourselves
else:  # First run: read docs/, split into chunks, embed and save
    documents = []
    for name in sorted(os.listdir(DOCS_DIR)):
        with open(os.path.join(DOCS_DIR, name), encoding='utf-8') as f:
            documents.append(Document(page_content=f.read(), metadata={'source': name}))

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MEMORY.md')


def load_memory() -> str:
    if not os.path.exists(MEMORY_FILE):
        return ''
    with open(MEMORY_FILE, encoding='utf-8') as f:
        return f.read().strip()


def with_memory(system_prompt: str) -> str:
    memory = load_memory()
    if not memory:
        return system_prompt
    return f'{system_prompt}\n\nLong-term memory (facts the user asked you to remember):\n{memory}'


class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'web', 'code', 'memory', 'forget'] = Field(..., description='Classify whether the user wants to just chat, ask question, change code in the project, have something remembered or have something forgotten.')


class DocumentGrade(BaseModel):
    relevant: bool = Field(..., description='True only if the documents answer the question.')


class ForgetFacts(BaseModel):
    line_numbers: list[int] = Field(..., description='Numbers of the memory lines to delete. Empty if nothing matches.')


class MemoryFact(BaseModel):
    fact: str | None = Field(None, description='The fact to remember, as one short self-contained sentence. None if there is nothing worth remembering.')


class State(TypedDict): # Like 'MessagesState'
    messages: Annotated[list, add_messages]
    message_intent: str | None
    next_node: str | None
    coding_request: str | None  # Rewritten instruction for Claude Code (kept out of messages)

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier, method='function_calling')

    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify if the user wants to chat ("chat"), retrieve knowledge about the company Zentrix Labs ("knowledge"), search the web for current or general information ("web"), change code ("code") explicitly asks you to remember something ("memory") or asks you to forget something ("forget").'},
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
    
    messages = [{'role': 'system', 'content': with_memory('You are a talkative chatbot for fun. Be nice.')}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_rag(state: State):
    print('Using rag')
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=3)

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    grade = llm.with_structured_output(DocumentGrade, method='function_calling').invoke(
        f'Question: {query}\n\nDocuments:\n{context}')
    if not grade.relevant:
        return {}  # Nothing added -> last message is still the user's -> fall back to web

    messages = [
        {'role': 'system', 'content': with_memory(f'You are a RAG agent. Answer the user using only the context below. If the answer is not in it, say you don\'t know.\n\nContext:\n{context}')},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_web(state: State):
    print('Using web')
    query = state['messages'][-1].content
    results = []
    for _ in range(3):  # Search engines often block / rate-limit scraping -> retry, then treat as no results
        try:
            results = DDGS().text(query, max_results=5)
            break
        except Exception:
            time.sleep(2)  # Wait before retrying

    context = '\n'.join(f'- {result["title"]}: {result["body"]}' for result in results)

    messages = [
        {'role': 'system', 'content': with_memory(f'Answer the user using only the web search results below. If the answer is not in them, say you don\'t know. Always start your answer with "(from web) ".\n\nSearch results:\n{context}')},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_code(state: State):
    print('Using code')
    user_prompt = state['coding_request']
    result = subprocess.run(
        ['claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'],
        cwd=WORKSPACE_DIR,
        capture_output=True,
        text=True
    )  

    output = result.stdout.strip() or result.stderr.strip()  
    
    return {'messages': [{'role': 'assistant', 'content': output}]}


def prepare_coding_request(state: State):
    messages = [
        {'role': 'system', 'content': with_memory('Rewrite the latest user coding request into a clear instruction for Claude Code. Use the conversation history as context. Only output the instruction, no explanation')}
    ] + state['messages']

    response = llm.invoke(messages)
    return {'coding_request': response.content}

def save_memory(state: State):
    print('Using memory')
    structured_llm = llm.with_structured_output(MemoryFact, method='function_calling')
    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Extract the fact the user wants remembered. Output one short, self-contained sentence.'},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])

    if not result.fact:
        return {'messages': [{'role': 'assistant', 'content': 'I found nothing to remember in that.'}]}

    fact = result.fact.strip()
    with open(MEMORY_FILE, 'a', encoding='utf-8') as f:
        f.write(f'- {fact}\n')

    return {'messages': [{'role': 'assistant', 'content': f'Saved to memory: {fact}'}]}


def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}


def auto_memory(state: State):
    structured_llm = llm.with_structured_output(MemoryFact, method='function_calling')
    result = structured_llm.invoke([
        {'role': 'system', 'content': f'Decide if the user message contains a lasting fact or preference about the user worth remembering for future conversations (e.g. name, job, preferences, projects). Ignore small talk, questions and temporary details. Do not repeat facts already in memory.\n\nMemory:\n{load_memory()}'},
        {'role': 'user', 'content': state['messages'][-2].content}  # The user's message (-1 is the chat answer)
    ])

    if result.fact:
        with open(MEMORY_FILE, 'a', encoding='utf-8') as f:
            f.write(f'- {result.fact.strip()}\n')
        print(f'\n[Remembered: {result.fact.strip()}]', end='')

    return {}


def forget_memory(state: State):
    print('Using forget')
    lines = load_memory().splitlines()
    numbered = '\n'.join(f'{number}. {line}' for number, line in enumerate(lines, 1))

    structured_llm = llm.with_structured_output(ForgetFacts, method='function_calling')
    result = structured_llm.invoke([
        {'role': 'system', 'content': 'The user wants you to forget something. Return the numbers of the memory lines that match the request.'},
        {'role': 'user', 'content': f'Request: {state["messages"][-1].content}\n\nMemory:\n{numbered}'}
    ])

    forgotten = [line for number, line in enumerate(lines, 1) if number in result.line_numbers]
    if not forgotten:
        return {'messages': [{'role': 'assistant', 'content': 'Nothing in memory matches that.'}]}

    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        f.writelines(f'{line}\n' for line in lines if line not in forgotten)

    return {'messages': [{'role': 'assistant', 'content': 'Forgot:\n' + '\n'.join(forgotten)}]}


graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('web_agent', prompt_llm_web)
graph_builder.add_node('coding_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding)
graph_builder.add_node('save_memory', save_memory)
graph_builder.add_node('forget_memory', forget_memory)
graph_builder.add_node('auto_memory', auto_memory)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('accept_coding', lambda state: 'end' if state.get('next_node') == 'denied' else state['next_node'], {'end': END, 'coding_agent': 'coding_agent', 'accept_coding': 'prepare_coding_request'})
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'web': 'web_agent', 'code': 'prepare_coding_request', 'memory': 'save_memory', 'forget': 'forget_memory'})
graph_builder.add_edge('prepare_coding_request', 'accept_coding')

graph_builder.add_edge('chat_agent', 'auto_memory')
graph_builder.add_edge('auto_memory', END)
graph_builder.add_conditional_edges('rag_agent', lambda state: 'web_agent' if state['messages'][-1].type == 'human' else END, ['web_agent', END])
graph_builder.add_edge('web_agent', END)
graph_builder.add_edge('coding_agent', END)
graph_builder.add_edge('save_memory', END)
graph_builder.add_edge('forget_memory', END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

config = {'configurable': {'thread_id': uuid.uuid4()}}

STREAMED_NODES = ('chat_agent', 'rag_agent', 'web_agent')  # Nodes whose answers are printed token by token

round_ = 1
while True:
    user_input = input(f'{round_} - Enter message: ')
    graph_input = {'messages': [{'role': 'user', 'content': user_input}]}
    streamed = False

    while graph_input:
        for token, metadata in graph.stream(graph_input, config=config, stream_mode='messages'):
            if metadata['langgraph_node'] in STREAMED_NODES:
                print(token.content, end='', flush=True)
                streamed = True

        state = graph.get_state(config)
        graph_input = None
        if state.interrupts:  # Coding agent asks for approval
            graph_input = Command(resume=input(f'{state.interrupts[0].value}\n> '))

    if streamed:
        print()  # End the streamed line
    else:
        print(state.values['messages'][-1].content)  # Coding, memory or denial reply
    print('\n')
    round_ += 1