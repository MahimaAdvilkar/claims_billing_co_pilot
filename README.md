# Claims & Billing Intelligence Co-Pilot
### Agentic RAG platform for healthcare revenue cycle analytics

Built to automate denial pattern detection, billing anomaly 
identification, and revenue recovery for healthcare billing teams.

🔗 [Live Demo](#) <!-- Add Streamlit link when deployed -->

---

## 🧠 What It Does

Healthcare billing teams spend hours manually identifying claim 
denials with no pattern visibility. This system automates that 
analysis end-to-end — ingesting claims data, detecting anomalies, 
benchmarking payers, and generating executive recovery recommendations.

---

## 🏗️ Architecture — 3-Tier Agent Graph

Built on LangGraph with 12 agents across 3 tiers:

| Tier | Agents | Execution |
|------|--------|-----------|
| **Tier 1** | `DataAgent` — loads CSV, filters by month, runs RAG retrieval | Sequential |
| **Tier 2** | `AnalysisAgent`, `TrendAnalysisAgent`, `MonthlySummaryAgent`, `ClientHistoryAgent`, `AnomalyAgent`, `PayerAnalyticsAgent`, `DenialReasonAgent`, `ClientProfitabilityAgent` | **Parallel** |
| **Tier 3** | `VisualizationAgent` → `ReportAgent` → `DecisionAgent` | Sequential synthesis |

Each agent writes to a distinct key in a shared TypedDict state 
(24 keys) — no race conditions.

---

## 🔄 How It Works
```
CSV / PDF Intake
      ↓
DataAgent (RAG retrieval via ChromaDB)
      ↓
┌─────────────────────────────────────┐
│  Parallel Analytics Agents (Tier 2) │
│  Anomaly · Denial · Payer · Trend   │
│  Summary · History · Profitability  │
└─────────────────────────────────────┘
      ↓
Visualization → Report → Decision
      ↓
Executive Summary + Recovery Actions
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Agent Orchestration | LangGraph |
| LLM | Ollama (llama3.1:8b) + HuggingFace fallback |
| Embeddings | nomic-embed-text (Ollama) + all-MiniLM-L6-v2 fallback |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| Analytics | Pandas + Altair |

---

## 🚀 Quick Start
```bash
# 1. Clone the repo
git clone https://github.com/MahimaAdvilkar/claims_billing_co_pilot
cd claims_billing_co_pilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (required for local LLM)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 4. Run Streamlit UI
streamlit run app_streamlit.py

# 5. Or run FastAPI backend
uvicorn app:app --reload
```

---

## 📊 What The Dashboard Shows

- Claims denial patterns by payer and reason code
- Month-over-month trend analysis
- Anomaly detection across billing data
- Client profitability breakdown
- Executive summary with recommended recovery actions
- Export to CSV and JSON

---

## 🏥 Domain Context

Built from real healthcare billing experience at Omatochi — 
where this system's design helped recover $300K+ in revenue 
across 500+ clients by identifying denial patterns and 
redesigning claims workflows.

---

## 📁 Project Structure
```
claims_billing_co_pilot/
├── agents/          # 12 analytics agents
├── graph.py         # LangGraph 3-tier workflow
├── app_streamlit.py # Streamlit UI
├── app.py           # FastAPI backend
├── vectordb.py      # ChromaDB vector store
├── retriever.py     # RAG retrieval layer
├── parse_csvs.py    # CSV ingestion
├── parse_pdfs.py    # PDF ingestion
└── chunk_and_embed.py # Chunking + embedding pipeline
```

---

## 🔮 Roadmap

- [ ] Deploy to Streamlit Cloud with demo dataset
- [ ] Add cloud LLM support (OpenAI / Claude fallback)
- [ ] Expand eval metrics (recall, NDCG, MRR)
- [ ] Add Docker configuration
- [ ] Expand dataset beyond Jan-Feb 2025

---

*Built by [Mahima Advilkar](https://linkedin.com/in/mahimaadvilkar)*
