# TrendAtlas

### Mapping how attention moves.

TrendAtlas is an end-to-end data and AI system for understanding how YouTube trending attention evolves across **time and markets**.

Instead of treating the trending page as a static leaderboard, TrendAtlas collects repeated snapshots across the **United States, Canada, United Kingdom, and India** to reconstruct how videos enter trending, gain or lose momentum, persist, spread across markets, and disappear.

It then uses machine learning to estimate whether a video will remain trending the next day and Gemini to generate explanations grounded only in computed evidence.

## 🔗 Live Project

**[Open the Live TrendAtlas Dashboard](https://aasthasheth180-ops-youtube-s3-gemini-pip-dashboard-trend-mxq9ji.streamlit.app/)**

> **Note:** The live demo is hosted on Streamlit Community Cloud and may go to sleep after a period of inactivity. If the app is sleeping, click **“Yes, get this app back up!”** and allow a short time for it to restart.

**Technical Article:** Coming soon

---

## Why I Built This

A trending page can tell us what is popular **right now**.

But a single snapshot cannot answer:

- When did a video enter trending?
- Is it gaining momentum or falling?
- How quickly is attention growing?
- How long does it remain trending?
- Does attention stay local or spread across markets?
- What kinds of content repeatedly appear?
- Can we predict whether that attention will survive tomorrow?

TrendAtlas turns repeated YouTube trending snapshots into historical trend intelligence designed to answer those questions.

---

## Architecture
(## Architecture

![TrendAtlas Architecture](<img width="1920" height="1080" alt="Trendatlas architecture diagram template" src="https://github.com/user-attachments/assets/3538af60-9e9f-4451-9ef8-0802a13f95f2" />
)




TrendAtlas follows six stages:

**Collect → Store → Understand → Predict → Explain → Present**

### 1. Collect
The YouTube Data API is sampled on a recurring schedule across:

`US · CA · GB · IN`

The Python ingestion pipeline validates and enriches each collection before storing it as a historical snapshot.

### 2. Store
Snapshots are stored in **Amazon S3** using timestamped, partitioned paths rather than overwriting previous observations.

This turns a changing trending board into historical data.

### 3. Understand
Historical snapshots are reconstructed into intelligence about:

- **Momentum** — how quickly attention is moving
- **Lifecycle** — how long videos remain trending
- **Content** — what categories and creators repeatedly appear
- **Markets** — how attention appears across collected regions

### 4. Predict
A supervised machine-learning pipeline estimates:

> **Will this video still be trending tomorrow?**

### 5. Explain
Computed analytics and model outputs are converted into a structured fact package before being sent to **Gemini 2.5 Flash**.

The AI layer is instructed to interpret only the supplied evidence rather than inventing unsupported metrics or explanations.

### 6. Present
A Streamlit application turns the pipeline into eight connected analytical chapters:

**Pulse · Momentum · Lifecycle · Content · Regions · Prediction · AI Brief · Data Health**

---

## From Snapshots to Trend Intelligence

The core idea behind TrendAtlas is simple:

> **One snapshot shows state. Repeated snapshots reveal movement.**

For the same video, repeated observations allow the system to reconstruct:

```text
Entry → Rise → Peak → Persistence → Decline → Exit
```

From these observations, TrendAtlas derives metrics including:

- rank movement
- view, like, and comment growth
- view / like / comment velocity
- first and last observation
- trending duration
- observation frequency
- peak rank
- re-entry behavior
- cross-market presence
- creator and category patterns

---

## Next-Day Persistence Prediction

TrendAtlas goes beyond retrospective analytics by predicting whether each video-region state will remain trending the following day.

### Model

**Random Forest Classifier**

The model uses only information available up to the prediction point, including:

- current and previous rank
- rank movement
- observations so far
- trending duration so far
- view and engagement growth
- velocity metrics
- channel statistics
- category
- market
- cross-market presence

### Temporal Evaluation

The model uses a **time-based train/test split** rather than randomly mixing historical and future observations.

This better reflects the real deployment problem: learning from the past to predict a future period.

### Model Performance

| Metric | Random Forest |
|---|---:|
| **ROC-AUC** | **0.871** |
| **F1 Score** | **0.743** |
| **Precision** | **0.794** |
| **Recall** | **0.697** |

A logistic regression model was also evaluated as a baseline, while the Random Forest produced the strongest overall performance.

---

## Grounded AI Interpretation

TrendAtlas does not send raw data to an LLM and ask it to discover arbitrary conclusions.

Instead:

```text
Historical Analytics
        +
ML Predictions
        ↓
Structured Fact Package
        ↓
Grounding Rules
        ↓
Gemini 2.5 Flash
        ↓
Grounded AI Brief
```

The grounding layer restricts the generated brief to evidence computed by the pipeline.

Among other safeguards, the AI is instructed to:

- use only supplied facts
- avoid invented metrics
- avoid unsupported causal claims
- distinguish predictions from observations
- interpret cross-market behavior only within the collected markets

---

## Data Quality & Reliability

TrendAtlas includes a dedicated Data Health layer because analytics are only useful when the underlying pipeline can be trusted.

The system monitors:

- snapshot freshness
- expected market coverage
- enrichment coverage
- exact video-region-snapshot duplicates
- recent pipeline audit results

The ingestion pipeline also includes validation and retry handling before historical snapshots are committed to storage.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Data Source** | YouTube Data API v3 |
| **Ingestion** | Python |
| **Automation** | GitHub Actions |
| **Storage** | Amazon S3 |
| **Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **AI Interpretation** | Gemini 2.5 Flash |
| **Visualization** | Streamlit, Plotly |
| **Cloud Integration** | AWS / Boto3 |
| **Version Control & CI** | Git, GitHub Actions |

---

## Repository Structure

```text
.
├── ai/                         # Fact-package construction and grounded Gemini interpretation
├── analytics/                  # Historical trend intelligence
├── models/                     # Trained persistence model and ML artifacts
├── .github/
│   └── workflows/              # Scheduled ingestion automation
├── fetch_to_s3.py              # YouTube ingestion → S3
├── build_history.py            # Historical dataset construction
├── dashboard_trend.py          # TrendAtlas Streamlit application
├── requirements.txt
└── README.md
```

---

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/aasthasheth180-ops/YouTube-S3-Gemini-Pipeline.git
cd YouTube-S3-Gemini-Pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file locally.

```text
YOUTUBE_API_KEY=
YOUTUBE_REGIONS=US,CA,GB,IN

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
S3_BUCKET=

GOOGLE_API_KEY=
```

Never commit `.env` or credentials to Git.

### 5. Start TrendAtlas

```bash
streamlit run dashboard_trend.py
```

---

## Limitations

TrendAtlas should be interpreted within the scope of its collected data.

- Analysis currently covers **US, CA, GB, and IN**.
- Trending status represents observations captured during scheduled collection intervals rather than continuous monitoring.
- Cross-market findings refer only to the markets collected by this project.
- Missing channel enrichment may affect some creator-level analysis.
- Prediction performance reflects the available historical collection window and may change as attention patterns evolve.
- Model predictions identify statistical patterns and should not be interpreted as causal explanations of why a video trends.

---

## What I Learned

Building TrendAtlas required connecting several problems that are often treated separately: reliable data ingestion, historical reconstruction, feature engineering, temporal machine-learning evaluation, AI grounding, data-quality monitoring, and analytical product design.

The biggest shift was moving from asking:

> **“What is trending?”**

to:

> **“How does attention move, how long does it survive, where does it travel, and what is likely to happen next?”**

---

## Author

**Aastha Sheth**

Built as an end-to-end data engineering, machine learning, and applied AI project.
