import streamlit as st
from src.agents.graph import build_graph
import pandas as pd
import altair as alt
import json

st.set_page_config(
    page_title="Claims & Billing Copilot – Agentic RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.section-card {
    padding: 25px;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid #e8e8e8;
    margin-bottom: 20px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.04);
}
.metric-card {
    padding: 20px;
    border-radius: 14px;
    background: #f8faff;
    border: 1px solid #e0e6f1;
    box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
    text-align: center;
}
.alert-box {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid;
}
.alert-anomaly { border-left-color: #d9534f; background-color: #fef5f5; }
.alert-warning { border-left-color: #f0ad4e; background-color: #fffbf0; }
.alert-alert { border-left-color: #5bc0de; background-color: #f0f8fb; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧾 Claims & Billing Copilot — Agentic RAG")
st.markdown("*Powered by AI agents for intelligent billing analysis*")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    show_raw_data = st.checkbox("Show Raw Data Context", value=False)
    export_format = st.selectbox("Export Format", ["JSON", "CSV"])

# Main query input
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        "Ask a question",
        key="user_query",
        placeholder="e.g., Show me October denials by payer, Compare Aetna vs BlueCross payments month-to-month, Which clients are most profitable?",
    )
with col2:
    run_btn = st.button("▶️ Analyze", use_container_width=True)

# Initialize session state
if "app" not in st.session_state:
    st.session_state.app = build_graph()

if run_btn and query:
    with st.spinner("🔄 Analyzing billing data with AI agents..."):
        try:
            # ========== Initialize ALL state keys ==========
            initial_state = {
                "query": query,
                "master_df": None,
                "full_df": None,
                "filtered_month": "",
                "rag_context": [],
                "contexts": [],
                "rag_hits": [],
                "analysis": {},
                "trend_analysis": {},
                "monthly_summary": [],
                "client_history": [],
                "anomalies": [],
                "payer_analytics": [],
                "denial_analysis": {},
                "client_profitability": [],
                "visualizations": [],
                "report_md": "",
                "decisions": [],
                "data_error": "",
            }
            result = st.session_state.app.invoke(initial_state)
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            result = {}

    # Extract all results AFTER result is defined
    report = result.get("report_md", "")
    decisions = result.get("decisions", [])
    analysis = result.get("analysis", {})
    monthly_summary = result.get("monthly_summary", [])
    client_history = result.get("client_history", [])
    payer_analytics = result.get("payer_analytics", [])
    anomalies = result.get("anomalies", [])
    denial_analysis = result.get("denial_analysis", {})
    client_profitability = result.get("client_profitability", [])
    visualizations = result.get("visualizations", [])
    trend_analysis = result.get("trend_analysis", {})

    # 🔍 Prefer explicit rag_hits, then rag_context, then contexts
    rag_hits = (
        result.get("rag_hits")
        or result.get("rag_context")
        or result.get("contexts")
        or []
    )

    st.markdown("---")

    # =====================
    # 📈 Annual Growth & Monthly Trends
    # =====================
    if trend_analysis:
        st.markdown("---")
        st.subheader("📈 Annual Growth & Monthly Trends")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Year-over-Year Growth",
            f"{trend_analysis.get('year_growth_pct', 0)}%",
        )
        col2.metric(
            "Peak Month",
            f"{trend_analysis['highest_month'].get('month', 'N/A')} (${trend_analysis['highest_month'].get('amount', 0):,.0f})",
        )
        col3.metric(
            "Lowest Month",
            f"{trend_analysis['lowest_month'].get('month', 'N/A')} (${trend_analysis['lowest_month'].get('amount', 0):,.0f})",
        )

        # Monthly trend table
        if trend_analysis.get("monthly_data"):
            st.write("**Monthly Progression:**")
            trend_df = pd.DataFrame(trend_analysis["monthly_data"])
            st.dataframe(
                trend_df[
                    [
                        "Month",
                        "Billed",
                        "Received",
                        "Collection_Rate",
                        "Billed_Change_Pct",
                        "Claim_Count",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # Trend chart - Billing
            trend_chart = (
                alt.Chart(trend_df)
                .mark_line(point=True, size=3, color="#1f77b4")
                .encode(
                    x=alt.X("Month:N", title="Month", sort=list(trend_df["Month"])),
                    y=alt.Y("Billed:Q", title="Amount Billed ($)"),
                    tooltip=["Month", "Billed", "Received", "Collection_Rate"],
                )
                .properties(height=400, title="Monthly Billing Trend")
            )

            st.altair_chart(trend_chart, use_container_width=True)

            # Collection rate trend
            collection_chart = (
                alt.Chart(trend_df)
                .mark_line(point=True, size=3, color="#2ca02c")
                .encode(
                    x=alt.X("Month:N", title="Month", sort=list(trend_df["Month"])),
                    y=alt.Y(
                        "Collection_Rate:Q",
                        title="Collection Rate (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    tooltip=["Month", "Collection_Rate", "Claim_Count"],
                )
                .properties(height=400, title="Monthly Collection Rate Trend")
            )

            st.altair_chart(collection_chart, use_container_width=True)

    st.markdown("---")

    # =====================
    # 📊 Key Metrics
    # =====================
    if analysis:
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Total Billed</h3>
                    <h2>${analysis.get('total_billed', 0):,.2f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Total Received</h3>
                    <h2>${analysis.get('total_received', 0):,.2f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Outstanding</h3>
                    <h2 style="color:#d9534f;">${analysis.get('total_variance', 0):,.2f}</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Collection Rate</h3>
                    <h2 style="color:#5cb85c;">{analysis.get('collection_rate', 0):.1f}%</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # =====================
    # 📋 Claims by Status / Payer
    # =====================
    if analysis and ("by_status" in analysis or "by_payer" in analysis):
        col1, col2 = st.columns(2)

        with col1:
            if "by_status" in analysis:
                st.subheader("📋 Claims by Status")
                status_data = []
                for status, data in analysis["by_status"].items():
                    status_data.append(
                        {
                            "Status": status,
                            "Count": int(data.get("Claim_ID", 0)),
                            "Amount": float(data.get("Amount_Billed", 0)),
                        }
                    )
                status_df = pd.DataFrame(status_data)
                st.dataframe(status_df, use_container_width=True)

        with col2:
            if "by_payer" in analysis:
                st.subheader("🏥 Claims by Payer")
                payer_data = []
                for payer, data in analysis["by_payer"].items():
                    payer_data.append(
                        {
                            "Payer": payer,
                            "Count": int(data.get("Claim_ID", 0)),
                            "Billed": float(data.get("Amount_Billed", 0)),
                            "Received": float(data.get("Amount_Received", 0)),
                        }
                    )
                payer_df = pd.DataFrame(payer_data)
                st.dataframe(payer_df, use_container_width=True)

    st.markdown("---")

    # =====================
    # 📈 AI-Generated Visualizations
    # =====================
    if visualizations:
        st.subheader("📈 AI-Generated Visualizations")

        for idx, viz in enumerate(visualizations):
            viz_type = viz.get("type")
            title = viz.get("title")
            data = viz.get("data")

            if not data:
                continue

            st.markdown(f"**{title}**")

            if viz_type == "bar":
                df_viz = pd.DataFrame(data)
                if viz.get("color"):
                    chart = (
                        alt.Chart(df_viz)
                        .mark_bar(cornerRadius=8)
                        .encode(
                            x=alt.X(f"{viz['x']}:N", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            color=alt.Color(
                                f"{viz['color']}:N", scale=alt.Scale(scheme="set2")
                            ),
                            tooltip=[viz["x"], viz["y"], viz["color"]],
                        )
                        .properties(height=400)
                    )
                else:
                    chart = (
                        alt.Chart(df_viz)
                        .mark_bar(cornerRadius=8)
                        .encode(
                            x=alt.X(f"{viz['x']}:N", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            tooltip=[viz["x"], viz["y"]],
                        )
                        .properties(height=400)
                    )
                st.altair_chart(chart, use_container_width=True)

            elif viz_type == "line":
                df_viz = pd.DataFrame(data)
                if viz.get("color"):
                    chart = (
                        alt.Chart(df_viz)
                        .mark_line(point=True, size=3)
                        .encode(
                            x=alt.X(f"{viz['x']}:N", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            color=alt.Color(
                                f"{viz['color']}:N",
                                scale=alt.Scale(scheme="viridis"),
                            ),
                            tooltip=[viz["x"], viz["y"], viz["color"]],
                        )
                        .properties(height=400)
                    )
                else:
                    chart = (
                        alt.Chart(df_viz)
                        .mark_line(point=True, size=3)
                        .encode(
                            x=alt.X(f"{viz['x']}:N", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            tooltip=[viz["x"], viz["y"]],
                        )
                        .properties(height=400)
                    )
                st.altair_chart(chart, use_container_width=True)

            elif viz_type == "area":
                df_viz = pd.DataFrame(data)
                if viz.get("color"):
                    chart = (
                        alt.Chart(df_viz)
                        .mark_area(opacity=0.7)
                        .encode(
                            x=alt.X(f"{viz['x']}:N", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            color=alt.Color(
                                f"{viz['color']}:N", scale=alt.Scale(scheme="set1")
                            ),
                            tooltip=[viz["x"], viz["y"], viz["color"]],
                        )
                        .properties(height=400)
                    )
                else:
                    chart = (
                        alt.Chart(df_viz)
                        .mark_area(opacity=0.7)
                        .encode(
                            x=alt.X(f"{viz['x']}:N", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            tooltip=[viz["x"], viz["y"]],
                        )
                        .properties(height=400)
                    )
                st.altair_chart(chart, use_container_width=True)

            elif viz_type == "pie":
                df_viz = pd.DataFrame(data)
                chart = (
                    alt.Chart(df_viz)
                    .mark_arc(innerRadius=60)
                    .encode(
                        theta=alt.Theta(f"{viz['value']}:Q"),
                        color=alt.Color(
                            f"{viz['label']}:N", scale=alt.Scale(scheme="category10")
                        ),
                        tooltip=[viz["label"], viz["value"]],
                    )
                    .properties(height=400)
                )
                st.altair_chart(chart, use_container_width=True)

            elif viz_type == "scatter":
                df_viz = pd.DataFrame(data)
                if viz.get("color"):
                    chart = (
                        alt.Chart(df_viz)
                        .mark_circle(size=100, opacity=0.7)
                        .encode(
                            x=alt.X(f"{viz['x']}:Q", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            color=alt.Color(
                                f"{viz['color']}:N",
                                scale=alt.Scale(scheme="tableau20"),
                            ),
                            tooltip=[viz["x"], viz["y"], viz["color"]],
                        )
                        .properties(height=400)
                    )
                else:
                    chart = (
                        alt.Chart(df_viz)
                        .mark_circle(size=100, opacity=0.7)
                        .encode(
                            x=alt.X(f"{viz['x']}:Q", title=viz["x"]),
                            y=alt.Y(f"{viz['y']}:Q", title=viz["y"]),
                            tooltip=[viz["x"], viz["y"]],
                        )
                        .properties(height=400)
                    )
                st.altair_chart(chart, use_container_width=True)

            elif viz_type == "table":
                df_viz = pd.DataFrame(data)
                st.dataframe(df_viz, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================
    # 📅 Monthly Breakdown
    # =====================
    if monthly_summary:
        st.subheader("📅 Monthly Breakdown (Jan-Dec)")
        monthly_df = pd.DataFrame(monthly_summary)

        # Format currency columns
        for col in ["Billed", "Submitted", "Received", "Variance"]:
            if col in monthly_df.columns:
                monthly_df[col] = monthly_df[col].apply(
                    lambda x: f"${x:,.2f}"
                )

        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================
    # 👥 Client History
    # =====================
    if client_history:
        st.subheader("👥 Repeat Clients (Most Active)")
        client_df = pd.DataFrame(client_history)

        for col in ["Total_Billed", "Total_Received", "Variance"]:
            if col in client_df.columns:
                client_df[col] = client_df[col].apply(
                    lambda x: f"${x:,.2f}"
                )

        st.dataframe(client_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================
    # 🏥 Payer Analytics
    # =====================
    if payer_analytics:
        st.subheader("🏥 Payer Performance Analytics")
        payer_df = pd.DataFrame(payer_analytics)
        st.dataframe(payer_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================
    # 🚫 Denial Analysis
    # =====================
    if denial_analysis:
        st.subheader("🚫 Denial Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Denied", denial_analysis.get("total_denied", 0))
        col2.metric("Denial Rate", f"{denial_analysis.get('denial_rate', 0)}%")
        col3.metric(
            "Denied Amount",
            f"${denial_analysis.get('total_denied_amount', 0):,.2f}",
        )

        if denial_analysis.get("reasons"):
            st.write("**Top Denial Reasons:**")
            st.dataframe(
                pd.DataFrame(denial_analysis["reasons"]),
                use_container_width=True,
            )

    st.markdown("---")

    # =====================
    # 💰 Client Profitability
    # =====================
    if client_profitability:
        st.subheader("💰 Client Profitability & Risk")
        profit_df = pd.DataFrame(client_profitability)
        st.dataframe(profit_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # =====================
    # 🔍 Anomalies
    # =====================
    if anomalies:
        st.subheader("🔍 Detected Anomalies")
        for anomaly in anomalies:
            if anomaly["type"] == "high_variance":
                st.warning(
                    f"⚠️ {anomaly['client']}: {anomaly['variance_pct']}% variance (${anomaly['variance']:,.2f})"
                )
            elif anomaly["type"] == "high_denial_rate":
                st.error(
                    f"❌ {anomaly['payer']}: {anomaly['denial_rate']}% denial rate ({anomaly['denied_count']} claims)"
                )
            elif anomaly["type"] == "slow_payment":
                st.info(
                    f"⏱️ {anomaly['month']}: {anomaly['avg_days']:.0f} days avg payment time"
                )

    st.markdown("---")

    # =====================
    # 🚨 Alerts
    # =====================
    if decisions:
        st.subheader("🚨 Alerts & Variance Flags")
        for alert in decisions:
            alert_type = alert.get("type", "alert").lower()
            message = alert.get("message", "")

            if alert_type == "anomaly":
                st.markdown(
                    f"""
                    <div class="alert-box alert-anomaly">
                        <strong>⚠️ ANOMALY:</strong> {message}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            elif alert_type == "warning":
                st.markdown(
                    f"""
                    <div class="alert-box alert-warning">
                        <strong>⚡ WARNING:</strong> {message}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="alert-box alert-alert">
                        <strong>ℹ️ ALERT:</strong> {message}
                    </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("✅ No alerts detected. Claims and billing data is healthy.")

    st.markdown("---")

    # =====================
    # 📘 Executive Summary
    # =====================
    st.subheader("📘 Executive Summary")
    if report:
        st.markdown(report)
    else:
        st.warning("Report generation in progress...")

    st.markdown("---")

    # =====================
    # 📄 RAG Context
    # =====================
    with st.expander("📄 View Raw Extracted Details (RAG Context)"):
        if not rag_hits:
            st.info("No additional context extracted.")
        else:
            for i, hit in enumerate(rag_hits, start=1):
                text = hit.get("text", "")
                meta = hit.get("metadata") or {}
                score = hit.get("score", None)

                st.markdown(f"**Chunk {i}**")
                if score is not None:
                    st.caption(f"Similarity score: {score:.3f}")

                if text:
                    st.write(text)

                if meta:
                    st.caption(f"Metadata: {meta}")

                st.markdown("---")

    # =====================
    # 💾 Export
    # =====================
    st.markdown("---")
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export as CSV"):
            if monthly_summary:
                csv = pd.DataFrame(monthly_summary).to_csv(index=False)
                st.download_button(
                    "Download Monthly Summary",
                    csv,
                    "monthly_summary.csv",
                    "text/csv",
                )

    with col2:
        if st.button("📥 Export as JSON"):
            json_data = json.dumps(
                {
                    "analysis": analysis,
                    "monthly_summary": monthly_summary,
                    "client_history": client_history,
                    "payer_analytics": payer_analytics,
                    "denial_analysis": denial_analysis,
                    "client_profitability": client_profitability,
                    "anomalies": anomalies,
                    "trend_analysis": trend_analysis,
                    "decisions": decisions,
                    "rag_hits": rag_hits,
                },
                indent=2,
                default=str,
            )
            st.download_button(
                "Download Full Report",
                json_data,
                "billing_report.json",
                "application/json",
            )

else:
    st.info("👉 Enter a query and click **Analyze** to get started!")

    st.markdown(
        """
    ### 💡 Example Queries:
    - "Show me May claims"
    - "Compare Aetna vs BlueCross payments"
    - "Which clients are most profitable?"
    - "Show October denials by payer"
    - "What's my monthly revenue trend?"
    - "Show payment delays by insurance company"
    """
    )
