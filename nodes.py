import time
import subprocess
from ddgs import DDGS # For the web search
from langgraph.types import interrupt

from core import llm, vector_store, with_memory, load_memory, MEMORY_FILE, WORKSPACE_DIR, State, IntentClassifier, DocumentGrade, ForgetFacts, MemoryFact


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier, method='function_calling')

    result = structured_llm.invoke([
        {'role': 'system', 'content': 'Determine / classify if the user wants to chat ("chat"), retrieve knowledge about the company Zentrix Labs ("knowledge"), search the web for current or general information ("web"), change code ("code"), asks you to store new information ("memory", e.g. "remember that ...") or asks you to forget something ("forget"). Questions about the user or about what you remember are "chat".'},
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
        {'role': 'system', 'content': 'Extract the fact the user wants remembered. Output one short, self-contained sentence. If the message contains no concrete fact to remember, return no fact.'},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])

    if not result.fact:
        return {'messages': [{'role': 'assistant', 'content': 'I found nothing to remember in that.'}]}

    fact = result.fact.strip()
    with open(MEMORY_FILE, 'a', encoding='utf-8') as f:
        f.write(f'- {fact}\n')

    return {'messages': [{'role': 'assistant', 'content': f'Saved to memory: {fact}'}]}


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
