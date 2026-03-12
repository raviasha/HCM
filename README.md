# HCM AI Insights — Prototype

An AI-powered Human Capital Management insights dashboard that analyzes employee attrition data and Voice of Employee feedback to generate actionable workforce analytics using GPT-4o.

## Architecture

```
┌─────────────────────┐      HTTP       ┌──────────────────────┐
│   Streamlit (8501)  │ ──────────────▶ │   FastAPI (8000)     │
│   - File upload     │                 │   - Data ingestion   │
│   - Dashboard UI    │ ◀────────────── │   - ChromaDB ops     │
│   - Plotly charts   │    JSON         │   - OpenAI GPT-4o    │
└─────────────────────┘                 └──────────────────────┘
                                              │         │
                                              ▼         ▼
                                        ┌─────────┐  ┌─────────┐
                                        │ ChromaDB│  │ OpenAI  │
                                        │ (local) │  │  API    │
                                        └─────────┘  └─────────┘
```

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

This creates synthetic datasets for 3 companies under `data/synthetic/`:
- **NovaTech Solutions** (Tech) — Engineering burnout, career stagnation
- **Meridian Retail Group** (Retail) — Low pay, Sales turnover
- **Pinnacle Healthcare** (Healthcare) — Work-life balance, clinical burnout

### 4. Run the Application

**Terminal 1 — API Server:**
```bash
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Streamlit Dashboard:**
```bash
source .venv/bin/activate
streamlit run app/dashboard.py --server.port 8501
```

### 5. Use the Dashboard

1. Open http://localhost:8501
2. **Quick demo:** Click a sample company button (NovaTech/Meridian/Pinnacle) in the sidebar, then click **Analyze**
3. **Custom upload:** Enter a company name, upload your CSV + JSON files, then click **Analyze**
4. Wait 30-60 seconds for AI analysis to complete
5. Explore the 4 dashboard tabs

## Demo Scenarios

| Company | Industry | What to Look For |
|---------|----------|-----------------|
| **NovaTech Solutions** | Tech | High Engineering attrition (45%+), overtime as top risk factor, feedback heavy on career stagnation and burnout |
| **Meridian Retail Group** | Retail | Extreme Sales turnover (58%+), low income as main driver, feedback focused on scheduling and compensation |
| **Pinnacle Healthcare** | Healthcare | Clinical & Nursing attrition (55%+), environment satisfaction issues, feedback about staffing shortages and burnout |

## Data Formats

### Attrition CSV (employees.csv)

| Column | Type | Description |
|--------|------|-------------|
| EmployeeID | int | Unique identifier |
| Age | int | Employee age |
| Department | string | Department name |
| JobRole | string | Job title |
| MonthlyIncome | int | Monthly salary |
| YearsAtCompany | int | Tenure |
| OverTime | Yes/No | Overtime status |
| JobSatisfaction | 1-4 | Satisfaction rating |
| Attrition | Yes/No | Left the company |
| ... | ... | 20+ additional HR fields |

### Voice of Employee JSON (feedback.json)

```json
[
  {
    "feedback_id": 1,
    "employee_id": 42,
    "feedback_type": "pulse_survey",
    "date": "2025-03-15",
    "question_prompt": "How are you feeling about your work?",
    "response_text": "The overtime culture is unsustainable...",
    "department": "Engineering"
  }
]
```

## Project Structure

```
HCM/
├── api/                    # FastAPI backend
│   ├── main.py             # App entry point, CORS, routers
│   ├── routes/
│   │   ├── attrition.py    # CSV upload & structured data queries
│   │   ├── feedback.py     # JSON upload & ChromaDB operations
│   │   └── insights.py     # AI insight generation pipeline
│   ├── services/
│   │   ├── chroma_service.py   # ChromaDB embedding & retrieval
│   │   ├── data_service.py     # Pandas backend (abstracted interface)
│   │   └── openai_service.py   # GPT-4o prompts for all AI tasks
│   └── models/
│       └── schemas.py      # Pydantic request/response models
├── app/                    # Streamlit frontend
│   ├── dashboard.py        # Main UI entry point
│   └── components/
│       ├── api_client.py   # HTTP client for FastAPI calls
│       ├── charts.py       # Plotly chart builders
│       └── kpi_cards.py    # KPI metric card helpers
├── config/
│   └── settings.py         # Centralized configuration
├── data/
│   ├── synthetic/          # Generated sample datasets
│   └── chroma_db/          # ChromaDB persistent storage (gitignored)
├── scripts/
│   └── generate_data.py    # Synthetic data generator
├── .env                    # API keys (gitignored)
├── requirements.txt
└── README.md
```

## Scalability Notes

The structured data service (`data_service.py`) uses an **abstracted `DataBackend` protocol**. The current `PandasBackend` works for up to ~100K rows. For production scale:

- Swap `PandasBackend` with `DuckDBBackend` (same interface, SQL-backed) for 100K–5M rows
- Change `DATA_BACKEND` in `config/settings.py`
- No changes needed to API routes or Streamlit frontend

## API Documentation

With the API running, visit http://localhost:8000/docs for interactive Swagger documentation.

## Cost Estimate

Each full analysis run makes ~5 GPT-4o API calls. Approximate cost per run: **$0.05–0.15** depending on data volume.
