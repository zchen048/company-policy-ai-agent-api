from langgraph.graph import StateGraph, START, END
from ..graph_states import GenGraphState
from ..node_functions import gen_functions

def build_gen_graph():
    graph_builder = StateGraph(GenGraphState)

    graph_builder.add_node("decide retrieval", gen_functions.decide_retrieve)
    graph_builder.add_node("retrieve documents", gen_functions.retrieve_policy)
    graph_builder.add_node("summarise documents", gen_functions.document_summary)
    graph_builder.add_node("check context length", gen_functions.check_context_length)
    graph_builder.add_node("truncate history", gen_functions.truncate_chat_history)
    graph_builder.add_node("generate answer", gen_functions.answer_user_query)

    graph_builder.add_edge(START, "decide retrieval")

    graph_builder.add_conditional_edges(
        "decide retrieval",
        gen_functions.need_retrieve,
        {
            True:"retrieve documents",
            False:"check context length"
        }
    )

    graph_builder.add_edge("retrieve documents", "summarise documents")
    graph_builder.add_edge("summarise documents", "check context length")

    graph_builder.add_conditional_edges(
        "check context length",
        gen_functions.context_length_conditional,
        {
            "generate":"generate answer",
            "truncate":"truncate history"
        }
    )

    graph_builder.add_edge("truncate history", "check context length")
    graph_builder.add_edge("generate answer", END)

    graph = graph_builder.compile()

    return graph

gen_graph = build_gen_graph()