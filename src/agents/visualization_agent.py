import pandas as pd
import json
import ollama
from .data_coordinator import DataCoordinator


class VisualizationAgent:
    """
    Generate visualization specs based on query intent.
    Uses LLM to suggest appropriate chart types.
    Falls back to default visualizations if LLM fails.
    """

    def run(self, state: dict) -> dict:
        # Prefer month-filtered df, but keep a copy of full_df for trends
        df = state.get("master_df")
        df_trend = state.get("full_df")

        analysis = state.get("analysis", {})
        query = state.get("query", "").lower()

        # If we don't have any data at all, bail out safely
        if df is None or len(df) == 0:
            print("⚠️ VisualizationAgent: No data available")
            return {"visualizations": []}

        # Ensure df_trend is usable – fall back to df if needed
        if df_trend is None or getattr(df_trend, "empty", False):
            df_trend = df

        try:
            # Get data summary
            data_summary = self._get_data_summary(df, analysis, query)

            # Try LLM for smart visualizations
            prompt = f"""You are a data visualization expert for healthcare billing.
Based on the user query, suggest 2-3 appropriate visualizations.

User Query: {query}

Data Summary:
{data_summary}

RULES:
1. Data months are ordered January to December
2. Use color differentiation for categories
3. For trends use line/area charts
4. For comparisons use bar charts
5. For distributions use pie charts

Respond ONLY with JSON array of visualization configs:
[
  {{"type": "line", "title": "...", "x_col": "Service_Month", "y_col": "Amount_Billed", "color_by": null, "agg": "sum"}},
  {{"type": "bar", "title": "...", "x_col": "Status", "y_col": "Amount_Billed", "color_by": "Status", "agg": "sum"}}
]

Valid types: bar, line, area, pie, scatter, table"""

            try:
                response = ollama.chat(
                    model="llama3.1:8b",
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response["message"]["content"].strip()

                # Extract JSON from any extra text
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]") + 1

                if start_idx >= 0 and end_idx > 0:
                    json_str = response_text[start_idx:end_idx]
                    viz_configs = json.loads(json_str)

                    visualizations = []
                    for config in viz_configs[:3]:
                        viz_type = config.get("type", "bar")

                        # Use full_df for trend charts, filtered df for others
                        if viz_type in ["line", "area"]:
                            viz_df = df_trend
                        else:
                            viz_df = df

                        viz = self._generate_visualization(viz_df, config)
                        if viz:
                            visualizations.append(viz)

                    if visualizations:
                        print(f"✅ LLM generated {len(visualizations)} visualizations")
                        return {"visualizations": visualizations}
            except Exception as e:
                print(f"⚠️ LLM visualization failed: {e}")

            # Fallback to default visualizations
            print("✅ Using default visualizations (LLM failed)")
            return {"visualizations": self._default_visualizations(df, df_trend)}

        except Exception as e:
            print(f"❌ VisualizationAgent Error: {str(e)}")
            return {"visualizations": []}

    def _get_data_summary(self, df: pd.DataFrame, analysis: dict, query: str) -> str:
        """Create data summary for LLM context"""
        return f"""
Query: {query}
Columns: {', '.join(df.columns.tolist())}
Records: {len(df)}
Total Billed: ${analysis.get('total_billed', 0):,.0f}
Total Received: ${analysis.get('total_received', 0):,.0f}
Collection Rate: {analysis.get('collection_rate', 0):.1f}%
Payers: {', '.join(df['Payer'].unique().tolist())}
Months: January to December
Status Values: {', '.join(df['Status'].unique().tolist())}"""

    def _generate_visualization(self, df: pd.DataFrame, config: dict) -> dict:
        """Convert config to visualization data"""
        try:
            # Guard against None/empty df to avoid 'NoneType' errors
            if df is None or getattr(df, "empty", False):
                return None

            viz_type = config.get("type", "bar")
            x_col = config.get("x_col")
            y_col = config.get("y_col")
            color_by = config.get("color_by")
            agg = config.get("agg", "sum")
            title = config.get("title", f"{viz_type.capitalize()} Chart")

            if not x_col or not y_col:
                return None

            if x_col not in df.columns or y_col not in df.columns:
                return None

            if viz_type == "bar":
                return self._create_bar_chart(df, x_col, y_col, color_by, agg, title)
            elif viz_type == "line":
                return self._create_line_chart(df, x_col, y_col, color_by, agg, title)
            elif viz_type == "area":
                return self._create_area_chart(df, x_col, y_col, color_by, agg, title)
            elif viz_type == "pie":
                return self._create_pie_chart(df, x_col, y_col, title)
            elif viz_type == "scatter":
                return self._create_scatter_chart(df, x_col, y_col, color_by, title)
            elif viz_type == "table":
                return self._create_table(df, x_col, y_col, title)

        except Exception as e:
            print(f"⚠️ Visualization generation error: {e}")
            return None

    def _create_bar_chart(self, df, x_col, y_col, color_by, agg, title):
        """Create bar chart data"""
        if color_by and color_by != x_col and color_by in df.columns:
            chart_df = df.groupby([x_col, color_by])[y_col].agg(agg).reset_index()
        else:
            chart_df = df.groupby(x_col)[y_col].agg(agg).reset_index()

        if x_col == "Service_Month":
            chart_df = DataCoordinator.sort_by_month(chart_df, x_col)

        return {
            "type": "bar",
            "title": title,
            "data": chart_df.to_dict("records"),
            "x": x_col,
            "y": y_col,
            "color": color_by,
        }

    def _create_line_chart(self, df, x_col, y_col, color_by, agg, title):
        """Create line chart data"""
        if color_by and color_by != x_col and color_by in df.columns:
            chart_df = df.groupby([x_col, color_by])[y_col].agg(agg).reset_index()
        else:
            chart_df = df.groupby(x_col)[y_col].agg(agg).reset_index()

        if x_col == "Service_Month":
            chart_df = DataCoordinator.sort_by_month(chart_df, x_col)

        return {
            "type": "line",
            "title": title,
            "data": chart_df.to_dict("records"),
            "x": x_col,
            "y": y_col,
            "color": color_by,
        }

    def _create_area_chart(self, df, x_col, y_col, color_by, agg, title):
        """Create area chart data"""
        if color_by and color_by != x_col and color_by in df.columns:
            chart_df = df.groupby([x_col, color_by])[y_col].agg(agg).reset_index()
        else:
            chart_df = df.groupby(x_col)[y_col].agg(agg).reset_index()

        if x_col == "Service_Month":
            chart_df = DataCoordinator.sort_by_month(chart_df, x_col)

        return {
            "type": "area",
            "title": title,
            "data": chart_df.to_dict("records"),
            "x": x_col,
            "y": y_col,
            "color": color_by,
        }

    def _create_pie_chart(self, df, x_col, y_col, title):
        """Create pie chart data"""
        chart_df = df.groupby(x_col)[y_col].sum().reset_index()
        chart_df = chart_df.sort_values(y_col, ascending=False).head(10)

        return {
            "type": "pie",
            "title": title,
            "data": chart_df.to_dict("records"),
            "label": x_col,
            "value": y_col,
        }

    def _create_scatter_chart(self, df, x_col, y_col, color_by, title):
        """Create scatter chart data"""
        if color_by and color_by in df.columns:
            data = df[[x_col, y_col, color_by]].to_dict("records")
        else:
            data = df[[x_col, y_col]].to_dict("records")

        return {
            "type": "scatter",
            "title": title,
            "data": data,
            "x": x_col,
            "y": y_col,
            "color": color_by if color_by and color_by in df.columns else None,
        }

    def _create_table(self, df, x_col, y_col, title):
        """Create table data"""
        if x_col in df.columns:
            table_df = df.groupby(x_col).agg(
                {
                    "Amount_Billed": "sum",
                    "Amount_Received": "sum",
                    "Claim_ID": "count",
                }
            ).reset_index()
            table_df.columns = [
                x_col,
                "Total_Billed",
                "Total_Received",
                "Claim_Count",
            ]
            table_df["Variance"] = (
                table_df["Total_Billed"] - table_df["Total_Received"]
            )
            table_df["Collection_Rate"] = (
                table_df["Total_Received"]
                / table_df["Total_Billed"]
                * 100
            ).round(2)

            if x_col == "Service_Month":
                table_df = DataCoordinator.sort_by_month(table_df, x_col)
        else:
            table_df = df[
                [
                    "Claim_ID",
                    "Client_Name",
                    "Service_Month",
                    "Amount_Billed",
                    "Amount_Received",
                    "Status",
                ]
            ].head(50)

        return {
            "type": "table",
            "title": title,
            "data": table_df.to_dict("records"),
        }

    def _default_visualizations(self, df: pd.DataFrame, df_trend: pd.DataFrame) -> list:
        """Fallback visualizations if LLM fails"""
        try:
            visualizations = []

            # Ensure df_trend exists
            if df_trend is None or getattr(df_trend, "empty", False):
                df_trend = df

            # 1. Monthly trend (use full_df / df_trend)
            monthly_df = df_trend.groupby("Service_Month")["Amount_Billed"].sum().reset_index()
            monthly_df = DataCoordinator.sort_by_month(monthly_df, "Service_Month")
            visualizations.append(
                {
                    "type": "line",
                    "title": "Monthly Billing Trend (Jan-Dec)",
                    "data": monthly_df.to_dict("records"),
                    "x": "Service_Month",
                    "y": "Amount_Billed",
                    "color": None,
                }
            )

            # 2. Status breakdown (use master_df)
            status_df = df.groupby("Status")["Amount_Billed"].sum().reset_index()
            visualizations.append(
                {
                    "type": "bar",
                    "title": "Claims by Status",
                    "data": status_df.to_dict("records"),
                    "x": "Status",
                    "y": "Amount_Billed",
                    "color": "Status",
                }
            )

            # 3. Payer comparison (use master_df)
            payer_df = (
                df.groupby("Payer")["Amount_Received"]
                .sum()
                .reset_index()
                .sort_values("Amount_Received", ascending=False)
                .head(8)
            )
            visualizations.append(
                {
                    "type": "bar",
                    "title": "Top Payers by Amount Received",
                    "data": payer_df.to_dict("records"),
                    "x": "Payer",
                    "y": "Amount_Received",
                    "color": "Payer",
                }
            )

            return visualizations

        except Exception as e:
            print(f"❌ Default visualization error: {e}")
            return []
