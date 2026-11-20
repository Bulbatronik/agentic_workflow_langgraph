from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from core import State
from nodes import classify_intent, prompt_llm_chat, prompt_llm_rag, prompt_llm_web, prompt_llm_code, prepare_coding_request, accept_coding, save_memory, forget_memory, auto_memory


def build_graph():
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

    return graph
