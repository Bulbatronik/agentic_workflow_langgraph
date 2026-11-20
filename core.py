import os
import warnings
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
warnings.filterwarnings('ignore', message='.*langchain-community', category=DeprecationWarning)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph.message import add_messages


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
