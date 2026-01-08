# filepath: /Users/mahimaadvilkar/Desktop/ALL ABOUT AI/Claims and Billing Co-Pilot/src/agents/graph.py

from langgraph.graph import StateGraph
from typing import TypedDict, Any

from .data_agent import DataAgent
from .analysis_agent import AnalysisAgent
from .trend_analysis_agent import TrendAnalysisAgent
from .monthly_summary_agent import MonthlySummaryAgent
from .client_history_agent import ClientHistoryAgent
from .anomaly_agent import AnomalyAgent
from .payer_analytics_agent import PayerAnalyticsAgent
from .denial_reason_agent import DenialReasonAgent
from .client_profitability_agent import ClientProfitabilityAgent
from .visualization_agent import VisualizationAgent
from .report_agent import ReportAgent
from .decision_agent import DecisionAgent


class State(TypedDict):
    """
    Shared state across all agents.
    Uses reducers for keys that multiple agents might update in parallel.
    """

    # Input / core data
    query: str
    master_df: Any
    full_df: Any
    filtered_month: str

    # RAG: vector hits / contexts
    rag_context: list[dict[str, Any]]
    contexts: list[dict[str, Any]]
    rag_hits: list[dict[str, Any]]

    # Tier 2: Analytics outputs (these don't conflict - each agent writes to unique key)
    analysis: dict
    trend_analysis: dict
    monthly_summary: list
    client_history: list
    anomalies: list
    payer_analytics: list
    denial_analysis: dict
    client_profitability: list

    # Tier 3: Synthesis outputs
    visualizations: list
    report_md: str
    decisions: list

    # Error handling
    data_error: str


def build_graph():
    """
    Build LangGraph with 3-tier execution.

    TIER 1 (Sequential):
    - data: Load CSV once

    TIER 2 (PARALLEL - 8 agents):
    - Each agent writes to unique state keys (no conflicts)
    - All run simultaneously after data loads

    TIER 3 (Sequential):
    - visualization, report, decision converge sequentially
    """

    g = StateGraph(State)

    # ========== TIER 1: Data Loading ==========
    g.add_node("data", DataAgent().run)

    # ========== TIER 2: Parallel Analytics ==========
    g.add_node("analysis", AnalysisAgent().run)
    g.add_node("trend_analysis", TrendAnalysisAgent().run)
    g.add_node("monthly_summary", MonthlySummaryAgent().run)
    g.add_node("client_history", ClientHistoryAgent().run)
    g.add_node("anomaly", AnomalyAgent().run)
    g.add_node("payer_analytics", PayerAnalyticsAgent().run)
    g.add_node("denial_reason", DenialReasonAgent().run)
    g.add_node("client_profitability", ClientProfitabilityAgent().run)

    # ========== TIER 3: Synthesis ==========
    g.add_node("visualization", VisualizationAgent().run)
    g.add_node("report", ReportAgent().run)
    g.add_node("decision", DecisionAgent().run)

    # ========== ENTRY POINT ==========
    g.set_entry_point("data")

    # ========== TIER 1 → TIER 2 (All parallel agents start after data) ==========
    tier2_agents = [
        "analysis",
        "trend_analysis",
        "monthly_summary",
        "client_history",
        "anomaly",
        "payer_analytics",
        "denial_reason",
        "client_profitability",
    ]

    for agent in tier2_agents:
        g.add_edge("data", agent)

    # ========== TIER 2 → TIER 3 (All converge to visualization) ==========
    # This creates a convergence point - LangGraph waits for all Tier 2 agents
    for agent in tier2_agents:
        g.add_edge(agent, "visualization")

    # ========== TIER 3: Sequential ==========
    g.add_edge("visualization", "report")
    g.add_edge("report", "decision")

    # ========== EXIT POINT ==========
    g.set_finish_point("decision")

    return g.compile()
