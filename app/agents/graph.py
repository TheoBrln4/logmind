from langgraph.graph import END, StateGraph

from app.agents.embed_agent import embed_agent
from app.agents.parser_agent import parser_agent
from app.agents.pattern_agent import pattern_agent
from app.agents.persist_agent import persist_agent
from app.agents.rca_agent import rca_agent
from app.agents.report_agent import report_agent
from app.agents.state import AnalysisState


def build_graph() -> StateGraph:
    """Assemble the LangGraph pipeline: parser → embed → pattern → rca → report → persist."""
    graph = StateGraph(AnalysisState)

    graph.add_node("parser", parser_agent)
    graph.add_node("embed", embed_agent)
    graph.add_node("pattern", pattern_agent)
    graph.add_node("rca", rca_agent)
    graph.add_node("report", report_agent)
    graph.add_node("persist", persist_agent)

    graph.set_entry_point("parser")
    graph.add_edge("parser", "embed")
    graph.add_edge("embed", "pattern")
    graph.add_edge("pattern", "rca")
    graph.add_edge("rca", "report")
    graph.add_edge("report", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
