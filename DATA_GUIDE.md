# Bank Marketing Dataset — Plain-English Guide

**What it is:** real data from a Portuguese bank's phone marketing campaigns.
Each row = one client that was called. The bank wants to know: will this
client subscribe to a term deposit?

- **Rows:** 45,211 clients
- **Target (`y`):** `yes` (5,289 / 11.7%) or `no` (39,922) — subscribed or not
- **Source:** UCI Machine Learning Repository (public, free)

## The columns, in plain English

**Who the client is**
- `age` — age in years
- `job` — type of job: admin, blue-collar, technician, management, retired, student, ...
- `marital` — married / single / divorced
- `education` — primary / secondary / tertiary / unknown

**Their money situation**
- `default` — do they have credit in default? (yes/no)
- `balance` — average yearly account balance in euros
- `housing` — do they have a housing loan? (yes/no)
- `loan` — do they have a personal loan? (yes/no)

**How the bank contacted them**
- `contact` — how: cellular or telephone
- `day` — day of the month they were last called
- `month` — month of last call (jan/feb/...)
- `duration` — how long the last call lasted, in seconds.
  ⚠️ **Important:** `duration` is only known *after* the call ends, so it
  can't be used to decide *who* to call. Using it is called **leakage** —
  the model cheating with information it wouldn't have in real life.
  We drop it (or keep it only to demonstrate the leakage lesson).
- `campaign` — how many times they were called in this campaign
- `pdays` — days since they were last called in a *previous* campaign
  (-1 means never called before)
- `previous` — how many times they were called before this campaign
- `poutcome` — what happened last campaign: success / failure / unknown

## Why this dataset is great for our project

1. **It's finance** — matches data-analyst job postings (customer analytics,
   campaign analysis).
2. **Questions write themselves** — "which job subscribes most?", "does
   balance predict subscription?", "which month worked best?" Perfect demo
   material for the AI assistant.
3. **Imbalanced target (11.7% yes)** — same lesson as before: accuracy lies,
   use AUC / F1. You already know this one.
4. **Leakage trap built in** — `duration` teaches one of the most important
   real-world ML lessons: a model with 0.95 AUC that is completely useless.

## Your first tasks (Phase 1)
1. Open `bank_data/bank-full.csv` and eyeball it.
2. Run: `src/phase1_eda_baseline.py` and read every printed line.
3. Answer in one sentence each: which job subscribes most? Does higher
   balance mean more subscriptions? What does `pdays = -1` mean?
4. Exercise: train once WITH `duration`, once WITHOUT. Compare AUC.
   The gap you see is the price of leakage — remember it forever.
