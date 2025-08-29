from langgraph.graph import StateGraph, START, END
from ..graph_states import DetailsGraphState
from ..node_functions import details_functions

def build_details_graph():
    graph_builder = StateGraph(DetailsGraphState)

    graph_builder.add_node("check intent", details_functions.classify_message_intent)
    graph_builder.add_node("context removal", details_functions.effective_context_removal)
    graph_builder.add_node("need details", details_functions.get_more_details)
    graph_builder.add_node("divert back", details_functions.divert_to_policy)

    graph_builder.add_edge(START, "check intent")
    graph_builder.add_conditional_edges(
        "check intent",
        details_functions.intent_conditional,
        {
            "get details":"need details",
            "remove":"context removal",
            "divert":"divert back",
            "end": END
        }
    )

    graph_builder.add_edge("context removal", "need details")
    graph_builder.add_edge("need details", END)
    graph_builder.add_edge("divert back", END)

    graph = graph_builder.compile()

    return graph

details_graph = build_details_graph()