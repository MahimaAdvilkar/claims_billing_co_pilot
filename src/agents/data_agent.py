import pandas as pd
from pathlib import Path
import re

# --- RAG integration ---
try:
    from src.rag.retriever import retrieve as rag_retrieve
    rag_available = True
except ImportError:
    rag_available = False


class DataAgent:
    """
    FOUNDATION AGENT: Load CSV data once, pass to all other agents.
    Also attaches RAG results (rag_context / rag_hits) into the state.
    """

    def run(self, state: dict) -> dict:
        try:
            # ---------------------------
            # Locate CSV
            # ---------------------------
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
            csv_files = list(data_dir.glob("*.csv"))

            if not csv_files:
                return {
                    "master_df": None,
                    "full_df": None,
                    "rag_context": [],
                    "contexts": [],
                    "rag_hits": [],
                    "data_error": f"No CSV found in {data_dir}",
                }

            csv_path = csv_files[0]
            print(f"📂 Loading CSV from: {csv_path}")
            df = pd.read_csv(csv_path)

            # ---------------------------
            # Validate required columns
            # ---------------------------
            required_cols = [
                "Claim_ID",
                "Service_Month",
                "Amount_Billed",
                "Amount_Received",
                "Status",
                "Payer",
                "Client_Name",
                "Days_To_Payment",
                "Amount_Submitted",
                "Denial_Reason",
            ]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                return {
                    "master_df": None,
                    "full_df": None,
                    "rag_context": [],
                    "contexts": [],
                    "rag_hits": [],
                    "data_error": f"Missing columns: {missing_cols}",
                }

            # ---------------------------
            # Fix numeric types
            # ---------------------------
            numeric_cols = [
                "Amount_Billed",
                "Amount_Received",
                "Amount_Submitted",
                "Days_To_Payment",
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Remove duplicate claim rows
            df = df.drop_duplicates(subset=["Claim_ID"], keep="first")
            df_all = df.copy()

            # ---------------------------
            # Filter by month in query
            # ---------------------------
            query = state.get("query", "").lower()
            match = re.search(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                query,
            )

            if match:
                month = match.group(1).capitalize()
                df = df[df["Service_Month"] == month]
                print(f"📅 Filtered to {month}: {len(df)} claims")

            # ---------------------------
            # RAG Context Retrieval
            # ---------------------------
            rag_context: list = []
            if rag_available:
                try:
                    rag_context = rag_retrieve(query) or []
                    print(f"🔍 RAG: retrieved {len(rag_context)} chunks")
                except Exception as e:
                    print(f"⚠️ RAG retrieval error: {e}")
                    rag_context = []

            print(f"✅ Loaded {len(df)} claims successfully")

            # We return rag_context under 3 keys so everyone can use it:
            # - rag_context  (raw)
            # - contexts     (for UI)
            # - rag_hits     (explicit list of hits)
            return {
                "master_df": df,
                "full_df": df_all,
                "rag_context": rag_context,
                "contexts": rag_context,
                "rag_hits": rag_context,
                "data_error": None,
            }

        except Exception as e:
            print(f"❌ DataAgent Error: {str(e)}")
            return {
                "master_df": None,
                "full_df": None,
                "rag_context": [],
                "contexts": [],
                "rag_hits": [],
                "data_error": str(e),
            }
