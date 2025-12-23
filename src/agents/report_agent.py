# filepath: src/agents/report_agent.py
import json
import textwrap
from typing import Any, Dict, List

import ollama


class ReportAgent:
    """
    Turn analytics + RAG hits into an executive summary.

    Inputs from state:
      - query: str
      - analysis: dict
      - trend_analysis: dict
      - monthly_summary: list[dict]
      - client_history: list[dict]
      - payer_analytics: list[dict]
      - denial_analysis: dict
      - client_profitability: list[dict]
      - anomalies: list[dict]
      - rag_hits / rag_context / contexts: list[dict]

    Output:
      - report_md: str  (markdown)
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "").strip()

        analysis = state.get("analysis", {}) or {}
        trend_analysis = state.get("trend_analysis", {}) or {}
        monthly_summary = state.get("monthly_summary", []) or []
        client_history = state.get("client_history", []) or []
        payer_analytics = state.get("payer_analytics", []) or []
        denial_analysis = state.get("denial_analysis", {}) or {}
        client_profitability = state.get("client_profitability", []) or []
        anomalies = state.get("anomalies", []) or []

        # 🔍 Prefer explicit rag_hits, then rag_context, then contexts
        rag_hits = (
            state.get("rag_hits")
            or state.get("rag_context")
            or state.get("contexts")
            or []
        )

        try:
            # ---------- 1. Build compact context ----------

            metrics_block = self._build_metrics_summary(analysis, trend_analysis)
            denials_block = self._build_denials_summary(denial_analysis)
            anomalies_block = self._build_anomalies_summary(anomalies)
            rag_block = self._build_rag_block(rag_hits)

            monthly_json = self._safe_json(monthly_summary)
            client_json = self._safe_json(client_history[:20])
            payer_json = self._safe_json(payer_analytics[:20])
            profit_json = self._safe_json(client_profitability[:20])

            # ---------- 2. LLM prompt ----------

            prompt = f"""
You are a senior healthcare revenue cycle and billing analyst.
Your job is to explain what's happening in the claims / billing data
and answer the user's question in clear, actionable language.

USER QUERY:
{query or "(no explicit question, just analyze the current slice of data)"}

HIGH-LEVEL METRICS:
{metrics_block}

DENIALS & RISK SUMMARY:
{denials_block}

ANOMALY FLAGS:
{anomalies_block}

MONTHLY SUMMARY (JSON, may be subset):
{monthly_json}

TOP CLIENT HISTORY (JSON, may be subset):
{client_json}

PAYER ANALYTICS (JSON, may be subset):
{payer_json}

CLIENT PROFITABILITY (JSON, may be subset):
{profit_json}

RAG CONTEXT: RELEVANT RAW CLAIM ROWS (from CSV):
{rag_block}

INSTRUCTIONS FOR YOUR ANSWER:

1. Start with a short title line like: "Executive Summary – April 2025 Claims" (or whatever month/scope applies).
2. Then use these headings in markdown:
   - **Overview**
   - **Key Findings**
   - **Denials & Risk**
   - **Recommended Actions (Billing & Follow-up)**
3. Under **Key Findings**, use bullet points with concrete numbers:
   - Total billed, total received, collection rate
   - Any spikes in denials or partial payments
   - Any patterns you see in specific payers or clients
4. Under **Denials & Risk**, explicitly mention clients and payers from the RAG rows.
   - For example: "Emily Clark has 3 denied claims with Humana (auth expired / not medically necessary / under review)..."
   - Call out denial reasons and financial impact where possible.
5. Under **Recommended Actions**, give 3–6 *specific* steps:
   - What to appeal
   - Which authorizations to check / renew
   - Which payer relationships or denial patterns to investigate
   - Any documentation or coding clean-up
6. DO NOT invent new clients, payers, or amounts not implied by the data.
7. Output ONLY valid markdown, no JSON.
"""

            try:
                response = ollama.chat(
                    model="llama3.1:8b",
                    messages=[{"role": "user", "content": textwrap.dedent(prompt).strip()}],
                )
                report_md = (
                    response.get("message", {})
                    .get("content", "")
                    .strip()
                )

                if not report_md:
                    raise ValueError("Empty report from LLM")

                print("✅ ReportAgent: Report generated via LLM")
                return {"report_md": report_md}

            except Exception as e:
                print(f"⚠️ ReportAgent LLM error: {e}")
                # Fallback to template-based summary
                fallback = self._fallback_report(
                    query=query,
                    analysis=analysis,
                    denial_analysis=denial_analysis,
                    anomalies=anomalies,
                    rag_hits=rag_hits,
                )
                return {"report_md": fallback}

        except Exception as e:
            print(f"❌ ReportAgent Error: {e}")
            # Last-resort text
            return {
                "report_md": f"### Executive Summary\n\nAn error occurred while generating the report: `{e}`"
            }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_metrics_summary(
        self, analysis: Dict[str, Any], trend: Dict[str, Any]
    ) -> str:
        total_billed = analysis.get("total_billed", 0)
        total_received = analysis.get("total_received", 0)
        collection_rate = analysis.get("collection_rate", 0)
        total_variance = analysis.get("total_variance", 0)

        year_growth = trend.get("year_growth_pct", 0)
        highest = trend.get("highest_month", {}) or {}
        lowest = trend.get("lowest_month", {}) or {}

        parts = [
            f"- Total billed: ${total_billed:,.2f}",
            f"- Total received: ${total_received:,.2f}",
            f"- Outstanding (variance): ${total_variance:,.2f}",
            f"- Collection rate: {collection_rate:.1f}%",
            f"- Year-over-year growth: {year_growth:.1f}%",
            f"- Peak month: {highest.get('month', 'N/A')} (${highest.get('amount', 0):,.2f})",
            f"- Lowest month: {lowest.get('month', 'N/A')} (${lowest.get('amount', 0):,.2f})",
        ]
        return "\n".join(parts)

    def _build_denials_summary(self, denial_analysis: Dict[str, Any]) -> str:
        if not denial_analysis:
            return "- No denial data available."

        total_denied = denial_analysis.get("total_denied", 0)
        denial_rate = denial_analysis.get("denial_rate", 0)
        total_denied_amount = denial_analysis.get("total_denied_amount", 0)
        reasons = denial_analysis.get("reasons", []) or []

        lines = [
            f"- Total denied claims: {total_denied}",
            f"- Denial rate: {denial_rate:.1f}%",
            f"- Denied amount: ${total_denied_amount:,.2f}",
        ]

        if reasons:
            top_reasons = ", ".join(
                f"{r.get('reason', 'Unknown')} ({r.get('count', 0)} claims)"
                for r in reasons[:5]
            )
            lines.append(f"- Top denial reasons: {top_reasons}")
        return "\n".join(lines)

    def _build_anomalies_summary(self, anomalies: List[Dict[str, Any]]) -> str:
        if not anomalies:
            return "- No anomalies flagged by the anomaly agent."

        lines = []
        for a in anomalies[:10]:
            t = a.get("type", "")
            if t == "high_variance":
                lines.append(
                    f"- High variance client: {a.get('client')} – "
                    f"{a.get('variance_pct')}% variance (${a.get('variance', 0):,.2f})"
                )
            elif t == "high_denial_rate":
                lines.append(
                    f"- High denial rate payer: {a.get('payer')} – "
                    f"{a.get('denial_rate')}% ({a.get('denied_count')} claims)"
                )
            elif t == "slow_payment":
                lines.append(
                    f"- Slow payment month: {a.get('month')} – "
                    f"avg {a.get('avg_days', 0):.0f} days to payment"
                )
            else:
                lines.append(f"- {a}")
        return "\n".join(lines)

    def _build_rag_block(self, rag_hits: List[Dict[str, Any]]) -> str:
        if not rag_hits:
            return "(No RAG rows retrieved for this query.)"

        lines = []
        for i, hit in enumerate(rag_hits[:8], start=1):
            text = hit.get("text", "")
            meta = hit.get("metadata") or {}
            score = hit.get("score", None)

            header = f"[Claim {i}]"
            if score is not None:
                header += f" (similarity: {score:.3f})"
            lines.append(header)

            if meta:
                lines.append(f"  Metadata: {meta}")

            if text:
                # Keep each row readable but not massive
                trimmed = text.strip()
                if len(trimmed) > 450:
                    trimmed = trimmed[:450] + "..."
                lines.append(f"  Row: {trimmed}")

            lines.append("")  # blank line between rows

        return "\n".join(lines).strip()

    def _safe_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj, indent=2, default=str)
        except Exception:
            return "<unserializable>"

    def _fallback_report(
        self,
        query: str,
        analysis: Dict[str, Any],
        denial_analysis: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        rag_hits: List[Dict[str, Any]],
    ) -> str:
        """Used if Ollama call fails."""
        total_billed = analysis.get("total_billed", 0)
        total_received = analysis.get("total_received", 0)
        collection_rate = analysis.get("collection_rate", 0)
        total_variance = analysis.get("total_variance", 0)

        total_denied = denial_analysis.get("total_denied", 0)
        denied_amount = denial_analysis.get("total_denied_amount", 0)

        lines = []
        lines.append("### Executive Summary (Fallback)")
        lines.append("")
        if query:
            lines.append(f"**Query:** {query}")
            lines.append("")

        lines.append("#### Overview")
        lines.append(
            f"- Total billed: **${total_billed:,.2f}**, "
            f"Total received: **${total_received:,.2f}**, "
            f"Collection rate: **{collection_rate:.1f}%**."
        )
        lines.append(
            f"- Outstanding variance: **${total_variance:,.2f}** across the filtered dataset."
        )
        lines.append(
            f"- Denials: **{total_denied}** claims denied, totaling **${denied_amount:,.2f}**."
        )
        lines.append("")

        if rag_hits:
            lines.append("#### Notable Claims from RAG Context")
            for hit in rag_hits[:5]:
                txt = hit.get("text", "").strip()
                if not txt:
                    continue
                meta = hit.get("metadata") or {}
                lines.append(f"- `{txt}`  \n  _Metadata: {meta}_")
            lines.append("")

        if anomalies:
            lines.append("#### Detected Anomalies")
            for a in anomalies[:5]:
                lines.append(f"- {a}")
            lines.append("")

        lines.append("#### Recommended Next Steps")
        lines.append(
            "- Review high-variance clients and confirm whether additional payments are expected or write-offs are needed."
        )
        lines.append(
            "- Investigate denial reasons and prepare appeals for high-dollar or recurring denial patterns."
        )
        lines.append(
            "- For any payer with consistently slow payments or high denials, consider a focused audit and payer-level meeting."
        )

        return "\n".join(lines)
