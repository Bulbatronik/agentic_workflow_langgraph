import uuid
from langgraph.types import Command

from graph import build_graph


graph = build_graph()


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