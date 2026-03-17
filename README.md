# HCM AI Insights — Prototype

An AI-powered Human Capital Management insights dashboard that analyzes **any** employee dataset and qualitative feedback to generate actionable workforce analytics using GPT-4o. The system dynamically detects your data schema, identifies the key target variable, and builds a tailored analysis — no templates, no manual configuration.

**Live Demo:** [Streamlit App](https://lielzkq3r4afegksabsm4o.streamlit.app/) · **API:** Hosted on [Render](https://hcm-433y.onrender.com/docs)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Zero-Config Schema Detection** | Upload any HR CSV — GPT-4o auto-detects columns, data types, and the primary target variable (binary or numeric) |
| **Dynamic Analysis Plans** | GPT-4o generates a custom analysis plan for each dataset — no hardcoded templates |
| **Adaptive Feedback Ingestion** | JSON feedback files with any key names are auto-mapped (text, ID, department, date fields detected automatically) |
| **Semantic Search (ChromaDB)** | Qualitative feedback is embedded in a vector database for targeted, department-level semantic retrieval |
| **11-Step AI Pipeline** | Schema analysis → plan generation → pandas execution → targeted feedback retrieval → sentiment → themes → correlation → executive summary → recommendations → dashboard spec |
| **Interactive Plotly Charts** | All visualizations are interactive — hover, zoom, pan, and download as PNG |
| **Conversational AI** | Multi-turn conversations in Ask AI and Explore tabs with full context memory |
| **Dynamic Tab Names** | Tab labels adapt to reflect what was actually analyzed in your data |
| **Works Across Industries** | Tech, retail, healthcare, finance — any HCM dataset works out of the box |

---

## Architecture

```
┌──────────────────────────┐      HTTP      ┌───────────────────────────┐
│  Streamlit Cloud (8501)  │ ─────────────▶ │  FastAPI on Render (8000) │
│  - File upload           │                │  - Data ingestion         │
│  - 7-tab dashboard       │ ◀───────────── │  - 11-step AI pipeline    │
│  - Plotly charts         │     JSON       │  - Dynamic schema detect  │
│  - Conversational AI     │                │  - ChromaDB semantic ops  │
└──────────────────────────┘                └───────────────────────────┘
                                                   │         │
                                                   ▼         ▼
                                             ┌─────────┐  ┌─────────┐
                                             │ ChromaDB│  │ OpenAI  │
                                             │ (local) │  │ GPT-4o  │
                                             └─────────┘  └─────────┘
```

### AI Pipeline (11 Steps)

```
CSV + JSON Upload
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  1. GPT-4o Schema Analysis — detect columns + target var    │
 │  2. GPT-4o Analysis Plan — generate custom analysis steps   │
 │  3. Pandas Execution — run plan against DataFrame           │
 │  4. GPT-4o Feedback Queries — formulate dept-level queries  │
 │  5. ChromaDB Retrieval — semantic search per department     │
 │  6. GPT-4o Sentiment Analysis — classify feedback sentiment │
 │  7. GPT-4o Theme Extraction — surface recurring topics      │
 │  8. GPT-4o Correlation — link quant metrics ↔ sentiment     │
 │  9. GPT-4o Executive Summary — C-suite narrative            │
 │ 10. GPT-4o Recommendations — prioritized action items       │
 │ 11. GPT-4o Dashboard Spec — generate chart/KPI definitions  │
 └─────────────────────────────────────────────────────────────┘
       │
       ▼
  Dynamic Dashboard (7 tabs)
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- An OpenAI API key

### 2. Setup

```bash
cd HCM

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 3. Generate Sample Data

```bash
python scripts/generate_data.py
```

This creates **structurally distinct** synthetic datasets for 3 companies under `data/synthetic/`:

| Company | Industry | CSV Columns | Target Variable | Feedback Keys |
|---------|----------|-------------|-----------------|---------------|
| **NovaTech Solutions** | Technology | 26 cols (EmployeeID, Age, Department, JobRole, MonthlyIncome, OverTime, EngagementScore, Attrition...) | Attrition (Yes/No) | feedback_id, response_text, department |
| **Meridian Retail Group** | Retail | 16 cols (StaffID, HourlyWage, ShiftType, StoreRegion, EngagementScore...) | EngagementScore (1–10) | id, answer, store_region |
| **Pinnacle Healthcare** | Healthcare | 16 cols (EmpID, Unit, ShiftPattern, WeeklyPatientLoad, BurnoutIndex...) | BurnoutIndex (1–10) | entry_id, comment, unit |

Each company has completely different CSV columns, JSON schemas, target variables, and feedback formats — demonstrating the app's zero-config adaptability.

### 4. Run the Application

**Terminal 1 — API Server:**
```bash
source .venv/bin/activate
python3 -m uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit Dashboard:**
```bash
source .venv/bin/activate
python3 -m streamlit run app/dashboard.py --server.port 8501
```

### 5. Use the Dashboard

1. Open http://localhost:8501
2. **Quick demo:** Click a sample company button (NovaTech / Meridian / Pinnacle) in the sidebar, then click **🚀 Analyze**
3. **Custom upload:** Enter a company name, upload your CSV + JSON files, then click **Analyze**
4. Wait 30–60 seconds for the AI pipeline to complete
5. Explore all 7 dashboard tabs

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **📋 Overview** | KPI cards + one representative chart from each analysis section |
| **📉 [Dynamic] Analysis** | Deep-dive into the detected target variable — tab name adapts (e.g., "Attrition Analysis", "Engagement Score Analysis") |
| **💬 [Dynamic] Feedback** | AI-classified sentiment by department + theme extraction with sample quotes |
| **🧠 AI Insights** | Executive summary, correlation insights, and prioritized recommendations (🔴 High / 🟡 Medium / 🟢 Low risk) |
| **💡 Ask AI** | Conversational Q&A with full context — ask follow-up questions, get text answers + auto-generated charts |
| **✏️ Explore & Customize** | View all charts and modify them via chat — change colors, types, filters, or create new visualizations |
| **📖 User Guide** | Built-in documentation with features, data format guide, sample datasets, and tips |

---

## Data Formats

The system **auto-detects** your data schema. No specific column names required.

### Structured CSV

Any employee-level CSV works. The AI detects:
- **Target variable** — binary (Yes/No, 0/1) or numeric (scores, indices)
- **Department column** — for group-by analysis
- **Numeric features** — for correlation and risk factor analysis
- **Categorical features** — for distribution breakdowns

### Qualitative Feedback JSON

An array of objects with any field names. The system auto-detects:
- **Text field** — the longest string field (response_text, answer, comment, etc.)
- **Department field** — department, unit, team, store_region, etc.
- **ID field** — feedback_id, id, entry_id, etc.
- **Date field** — date, submitted_at, timestamp, etc.

Example (field names are flexible):
```json
[
  {
    "feedback_id": "F001",
    "department": "Engineering",
    "response_text": "I feel overworked and undervalued...",
    "feedback_type": "exit_interview"
  }
]
```

---

## Project Structure

```
HCM/
├── api/                        # FastAPI backend
│   ├── main.py                 # App entry, CORS, routers (v0.2.0)
│   ├── routes/
│   │   ├── data.py             # POST /api/data/upload-csv
│   │   ├── feedback.py         # POST /api/data/upload-json
│   │   └── insights.py         # POST /api/insights/generate, /ask
│   ├── services/
│   │   ├── chroma_service.py   # ChromaDB ingestion (auto-detects JSON keys) & semantic search
│   │   ├── data_service.py     # PandasBackend — dynamic schema, target detection, analysis execution
│   │   └── openai_service.py   # All GPT-4o calls (10 prompt types, loaded from YAML)
│   └── models/
│       └── schemas.py          # Pydantic models (InsightResponse, AnalysisSection, ChartSpec, etc.)
├── app/                        # Streamlit frontend
│   ├── dashboard.py            # Main UI — 7 dynamic tabs, @st.fragment for chat tabs
│   └── components/
│       ├── api_client.py       # HTTP client for FastAPI calls
│       ├── charts.py           # (Deprecated — replaced by dynamic rendering)
│       └── kpi_cards.py        # (Deprecated — replaced by dynamic rendering)
├── config/
│   ├── settings.py             # Centralized config (OpenAI, ChromaDB, ports)
│   └── prompts.yaml            # All 10 GPT-4o system prompts (editable without code changes)
├── data/
│   ├── synthetic/              # 3 structurally distinct sample datasets
│   │   ├── novatech/           # Tech — 26 CSV cols, Attrition binary target
│   │   ├── meridian/           # Retail — 16 CSV cols, EngagementScore numeric target
│   │   └── pinnacle/           # Healthcare — 16 CSV cols, BurnoutIndex numeric target
│   └── chroma_db/              # ChromaDB persistent storage (gitignored)
├── scripts/
│   └── generate_data.py        # Synthetic data generator (3 independent company generators)
├── .env                        # API keys (gitignored)
├── .streamlit/
│   └── secrets.toml            # Streamlit secrets — API_BASE_URL (gitignored)
├── render.yaml                 # Render deployment config
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Deployment

### Backend (Render)
- Configured via `render.yaml`
- Set `OPENAI_API_KEY` in the Render dashboard environment variables
- Auto-deploys from `main` branch

### Frontend (Streamlit Cloud)
- Points to the GitHub repo, runs `app/dashboard.py`
- Set `API_BASE_URL = "https://hcm-433y.onrender.com"` in Streamlit Cloud Secrets

### Local Development
- Backend: `python3 -m uvicorn api.main:app --reload --port 8000`
- Frontend: `python3 -m streamlit run app/dashboard.py --server.port 8501`
- `.streamlit/secrets.toml` should contain `API_BASE_URL = "http://localhost:8000"`

---

## Scalability Notes

The structured data service (`data_service.py`) uses an **abstracted `DataBackend` protocol**. The current `PandasBackend` works for up to ~100K rows. For production scale:

- Swap `PandasBackend` with `DuckDBBackend` (same interface, SQL-backed) for 100K–5M rows
- Change `DATA_BACKEND` in `config/settings.py`
- No changes needed to API routes or Streamlit frontend

## API Documentation

With the API running, visit http://localhost:8000/docs (or https://hcm-433y.onrender.com/docs) for interactive Swagger documentation.

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/data/upload-csv` | Upload structured employee CSV |
| POST | `/api/data/upload-json` | Upload qualitative feedback JSON |
| POST | `/api/insights/generate` | Run full 11-step AI pipeline |
| POST | `/api/insights/ask` | Conversational Q&A with context |
| GET | `/api/insights/status` | Check if data/insights exist |
| GET | `/api/insights/cached` | Retrieve previously generated insights |
| GET | `/health` | Health check |

## Cost Estimate

Each full analysis run makes ~8 GPT-4o API calls + embedding calls for feedback ingestion. Approximate cost per run: **$0.10–0.25** depending on data volume and feedback size.

---

*Built with FastAPI · Streamlit · GPT-4o · ChromaDB · Plotly*
