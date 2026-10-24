"""Phase 3: The AI Brain — ask the bank data questions in plain English.

HOW IT WORKS (the big picture, in 3 steps):
  1. You type a question like "which job subscribes most?"
  2. Gemini (Google's free AI) writes pandas code to answer it.
  3. We run that code in a SAFE SANDBOX and show you the answer (+ chart).

WHY A SANDBOX? The AI writes code, and we should never run AI-written code
blindly — it could delete files or do something nasty by mistake. The sandbox
below only lets the code touch the dataframe `df` and make charts. Everything
dangerous (files, internet, system commands) is blocked before the code runs.

RUN IT:
  1. Get a free API key: https://aistudio.google.com -> "Get API key"
     -> "Create API key". (No credit card. Takes ~1 minute.)
  2. Save the key where the script can find it (do NOT paste it into the code):
       Windows (PowerShell):  $env:GEMINI_API_KEY = "paste-your-key-here"
       (that lasts for the terminal session; use setx for permanent)
  3. Install what you need:  pip install google-genai pandas matplotlib
  4. Run:  python src/phase3_gemini_analyst.py
     Then type questions, e.g.:
       - which job has the highest subscription rate?
       - plot subscription rate by month
       - do people with higher balance subscribe more?

Phase 3 learning goals: APIs (talking to Gemini over the internet),
prompt engineering (telling the AI exactly what code to write),
and safe code execution (the sandbox pattern real AI agents use).
"""
import concurrent.futures
import os
import re

import matplotlib

matplotlib.use("Agg")  # no pop-up windows; charts save to files instead
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL = "gemini-3.8-flash"  # free-tier model, fast and good enough for code gen (2.5-flash retired for new users, Oct 2026)
# Fallbacks: if the main model is overloaded (503) or retired (404),
# try these next instead of failing. 2.0-flash is the oldest stable Flash
# and usually has spare capacity; 3.5-flash is a newer fallback.
FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-3.5-flash"]
CODE_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# 1. THE SANDBOX — what AI-written code is allowed to do
# ---------------------------------------------------------------------------
# BLOCKED_PATTERNS: anything matching these is refused BEFORE running.
# We block: file access, system commands, network, and Python's escape hatches
# (things like __import__ or getattr that could sneak around the other rules).
BLOCKED_PATTERNS = [
    r"__import__", r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bsubprocess\b",
    r"\bos\b", r"\bsys\b", r"\bopen\s*\(", r"\bexec\s*\(", r"\beval\s*\(",
    r"\bcompile\s*\(", r"\bglobals\s*\(", r"\blocals\s*\(", r"\bgetattr\s*\(",
    r"\bsetattr\s*\(", r"\bdelattr\s*\(", r"\binput\s*\(", r"\brequests\b",
    r"\bsocket\b", r"\burllib\b", r"\bshutil\b", r"\bpathlib\b",
]

# SAFE_BUILTINS: the only built-in functions the generated code may use.
# Notice what's missing on purpose: open, exec, eval, __import__, input.
SAFE_BUILTINS = {
    "len": len, "range": range, "list": list, "dict": dict, "tuple": tuple,
    "set": set, "str": str, "int": int, "float": float, "bool": bool,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "sorted": sorted, "enumerate": enumerate, "zip": zip, "print": print,
}


def safety_scan(code: str) -> str | None:
    """Return the reason the code is unsafe, or None if it looks fine."""
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, code):
            return f"blocked pattern: {pat}"
    if "answer" not in code:
        return "code must assign the result to a variable named `answer`"
    return None


def run_in_sandbox(code: str, df: pd.DataFrame):
    """Run AI-generated code with restricted powers and a time limit.

    Returns (ok, result_or_error). The code can only see: df, pd, np, plt
    and the safe builtins above. It runs in a worker thread so we can stop
    waiting if it hangs (timeout = CODE_TIMEOUT_SECONDS).
    """
    problem = safety_scan(code)
    if problem:
        return False, f"SAFETY BLOCK: {problem}"

    namespace = {
        "pd": pd, "np": np, "plt": plt, "df": df,
        "__builtins__": SAFE_BUILTINS,
    }

    def _target():
        exec(code, namespace)  # noqa: S102 - sandboxed on purpose
        return namespace.get("answer", None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_target)
        try:
            result = future.result(timeout=CODE_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            return False, f"timed out after {CODE_TIMEOUT_SECONDS}s (infinite loop?)"
        except Exception as e:  # noqa: BLE001 - show the AI's error to the user
            return False, f"code raised {type(e).__name__}: {e}"
    return True, result


# ---------------------------------------------------------------------------
# 2. THE PROMPT — telling Gemini exactly what code to write
# ---------------------------------------------------------------------------
# This is "prompt engineering": instead of chatting, we give the model a
# strict contract: data schema + rules + "code only". The schema (column
# names and types) is the key ingredient — without it the model guesses
# column names and the code breaks.
def build_prompt(question: str, df: pd.DataFrame) -> str:
    schema = "\n".join(f"- {c} ({df[c].dtype})" for c in df.columns)
    return f"""You are a pandas code generator. A DataFrame named `df` is already loaded.

Columns:
{schema}

Target column: `y` ("yes"/"no") — subscription to a term deposit.
Note: `duration` is known only AFTER a call ends, so never use it to explain subscriptions (data leakage).

Rules:
1. Output ONLY valid Python code. No markdown fences, no explanations, no comments.
2. You may use only: df, pd, np, plt, and basic builtins (len, round, sorted, ...).
3. Assign the final result (a value, Series, or DataFrame) to a variable named `answer`.
4. NEVER use: import statements, open(), exec, eval, os, sys, subprocess, network calls.
5. If the question asks for a chart/plot/visual/graph: build it with plt, save it with plt.savefig("chart.png"), then plt.close(). Still assign `answer`.
6. Prefer simple groupby/aggregation. Round floats to 3 decimals.

Question: {question}
Code:"""


def ask_gemini(question: str, df: pd.DataFrame) -> str:
    """Send the prompt to Gemini and return the raw code text it writes.

    Tries MODEL first, then FALLBACK_MODELS, if a model is overloaded
    (503) or retired (404). Auth errors (bad key) fail immediately —
    retrying those would be pointless.
    """
    from google import genai  # lazy import: sandbox tests run without the package

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com (Get API key -> Create API key), then:\n"
            '  PowerShell: $env:GEMINI_API_KEY = "your-key-here"'
        )
    client = genai.Client(api_key=api_key)
    prompt = build_prompt(question, df)
    last_error = None
    for model in [MODEL, *FALLBACK_MODELS]:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            break
        except Exception as e:  # noqa: BLE001 - 503/404 fall through to next model
            last_error = e
            if "503" in str(e) or "404" in str(e):
                continue
            raise
    else:
        raise last_error
    code = response.text.strip()
    # Models sometimes wrap code in ```python fences despite the instruction.
    code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


# ---------------------------------------------------------------------------
# 3. MAIN LOOP — ask, generate, sandbox, show
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv("data/bank_data/bank-full.csv", sep=";")
    print(f"Loaded bank data: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    print("Ask questions in plain English (type 'quit' to exit).\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        try:
            code = ask_gemini(question, df)
        except Exception as e:  # noqa: BLE001
            print(f"Gemini error: {e}\n")
            continue

        print("\n--- code Gemini wrote ---")
        print(code)
        print("-------------------------\n")

        ok, result = run_in_sandbox(code, df)
        if ok:
            print("Answer:")
            print(result)
            if os.path.exists("chart.png"):
                print("\nChart saved to chart.png")
                os.remove("chart.png")  # fresh chart per question
        else:
            print(result)
        print()


if __name__ == "__main__":
    main()
