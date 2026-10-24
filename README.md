# AI Data Analyst Assistant

**Upload any CSV. Ask a question in plain English. The AI writes the pandas code, runs it in a safe sandbox, and answers — with charts and explanations.**

> 🔗 **Live demo:** https://ai-data-analyst-assistant-dkwb6gas7yfupvwodmxjwr.streamlit.app

Built on the UCI Bank Marketing dataset (45,211 records of a bank's term-deposit marketing campaign), but it works with any CSV you upload.

## What it does

1. **Ask anything** — e.g. "which job has the highest subscription rate?" or "plot subscription rate by month"
2. **See the code** — Gemini writes real pandas code, shown to you before it runs
3. **Get the answer + chart** — the code runs in a restricted sandbox and the result (and chart) is displayed
4. **Ask "Why?"** — pick any row and get a SHAP explanation of the model's prediction: which features pushed toward "will subscribe" and which pushed away

## How it works

```
Your question (plain English)
        │
        ▼
Gemini (prompt engineering: column schema + strict "code only" contract)
        │
        ▼
Generated pandas code ──► Safety sandbox ──► Answer + chart
                              • blocklist: os, sys, open, eval, network...
                              • only df, pd, np, plt + safe builtins
                              • 30-second timeout
                              • result must be assigned to `answer`
```

**Model fallback:** if the main model (`gemini-3.8-flash`) is overloaded (503) or retired (404), the app automatically retries with `gemini-2.0-flash`, then `gemini-3.5-flash`.

## Findings from the bank data

- **Students subscribe at the highest rate** of any job group
- **Subscription spikes by month:** March 52%, December 47%, September 47%, October 44% — vs. May 7% and July 9%
- **Subscribers have higher average balances:** ~1,804 vs ~1,304 for non-subscribers
- **SHAP example (row 0):** model predicts "will not subscribe" (4.1% probability) — driven mostly by `contact_unknown` pushing against subscription, partly offset by tertiary education pushing toward it

## Phase 1 baselines (for reference)

| Model | AUC |
|---|---|
| Logistic Regression | 0.77 |
| HistGradientBoosting | 0.80 |

**Leakage lesson:** `duration` (call length) is only known *after* a call ends, so it was dropped — using it would leak the answer.

## Run it locally

```bash
pip install -r requirements.txt

# PowerShell (Windows)
$env:GEMINI_API_KEY = "your-key-here"   # free key from https://aistudio.google.com

# Command Prompt (Windows)
set GEMINI_API_KEY=your-key-here

# Terminal version (Phase 3)
python src/phase3_gemini_analyst.py

# Web app version (Phase 4)
python -m streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the repo, branch `main`, file `app.py`
3. **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. **Deploy**

## Project structure

```
app.py                          # Phase 4: Streamlit web app
src/
  phase1_eda_baseline.py        # Phase 1: EDA + baseline models + leakage exercise
  phase3_gemini_analyst.py      # Phase 3: Gemini code generation + safety sandbox
data/bank_data/bank-full.csv    # UCI Bank Marketing dataset (semicolon-separated)
```

## Tech stack

Python · pandas · Streamlit · Google Gemini API · SHAP · scikit-learn · matplotlib
