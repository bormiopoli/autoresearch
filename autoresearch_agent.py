#!/usr/bin/env python3
"""
Local Ollama autoresearch agent.

Loop:
    program.md + train.py + history
              ↓
           Ollama
              ↓
        proposed edit
              ↓
         git checkpoint
              ↓
        python train.py
              ↓
           metric
              ↓
        keep / revert
              ↓
            repeat

Requirements:
    pip install requests
    ollama must be running locally
    git repository recommended

Run:
    python autoresearch_agent.py

Environment variables:
    OLLAMA_MODEL=qwen3-coder:30b
    OLLAMA_URL=http://localhost:11434
    TRAIN_COMMAND="python train.py"
    METRIC_NAME=val_loss
    METRIC_MODE=min
    MAX_EXPERIMENTS=20
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

TRAIN_COMMAND = os.getenv(
    "TRAIN_COMMAND",
    "python train.py",
)

PROGRAM_FILE = Path("program.md")
TRAIN_FILE = Path("train.py")
RESULTS_FILE = Path("results.jsonl")

METRIC_NAME = os.getenv(
    "METRIC_NAME",
    "val_loss",
)

METRIC_MODE = os.getenv(
    "METRIC_MODE",
    "min",
).lower()

MAX_EXPERIMENTS = int(
    os.getenv(
        "MAX_EXPERIMENTS",
        "20",
    )
)

EXPERIMENT_TIMEOUT = int(
    os.getenv(
        "EXPERIMENT_TIMEOUT",
        "3600",
    )
)

OLLAMA_TIMEOUT = int(
    os.getenv(
        "OLLAMA_TIMEOUT",
        "600",
    )
)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def die(message):
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def run_command(command, timeout=None):
    print(f"\n$ {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        return 124, output + "\n[TIMEOUT]"

    print(result.stdout)
    return result.returncode, result.stdout


def read_text(path):
    if not path.exists():
        die(f"Missing file: {path}")

    return path.read_text(encoding="utf-8")


def write_text(path, text):
    path.write_text(
        text,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def git_available():
    code, _ = run_command(
        "git rev-parse --is-inside-work-tree"
    )
    return code == 0


def git_status():
    code, output = run_command(
        "git status --short"
    )

    if code != 0:
        die("Could not get git status.")

    return output


def git_checkpoint():
    """
    Create a temporary commit containing the current experiment state.

    We use a commit rather than git stash because it gives us a very
    reliable rollback point.
    """

    code, _ = run_command(
        "git add train.py && "
        'git commit -m "autoresearch experiment checkpoint" '
        "--no-verify"
    )

    return code == 0


def git_revert_last_commit():
    """
    Undo the checkpoint commit while preserving the previous state.
    """

    code, _ = run_command(
        "git reset --hard HEAD~1"
    )

    return code == 0


def git_commit_experiment(number, metric):
    message = (
        f"autoresearch: experiment {number}, "
        f"{METRIC_NAME}={metric}"
    )

    code, _ = run_command(
        "git add train.py && "
        f'git commit -m "{message}" --no-verify'
    )

    return code == 0


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_chat(messages):
    """
    Talk to the local Ollama server.

    We intentionally request plain JSON rather than relying on
    model-specific tool calling support.
    """

    url = OLLAMA_URL.rstrip("/") + "/api/chat"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as exc:
        die(
            f"Could not connect to Ollama at "
            f"{OLLAMA_URL}: {exc}"
        )

    if response.status_code != 200:
        die(
            f"Ollama returned HTTP {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    try:
        content = data["message"]["content"]
    except KeyError:
        die(f"Unexpected Ollama response:\n{data}")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        die(
            "Ollama did not return valid JSON.\n\n"
            f"Response:\n{content}"
        )


# ---------------------------------------------------------------------------
# Metric handling
# ---------------------------------------------------------------------------

def extract_metric(output):
    """
    Accept examples:

        val_loss: 1.234
        val_loss=1.234
        FINAL val_loss: 1.234

    Also accepts scientific notation.
    """

    pattern = (
        rf"{re.escape(METRIC_NAME)}"
        rf"\s*[:=]\s*"
        rf"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )

    matches = re.findall(
        pattern,
        output,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    return float(matches[-1])


def is_better(new, old):
    if old is None:
        return True

    if METRIC_MODE == "max":
        return new > old

    return new < old


# ---------------------------------------------------------------------------
# Experiment history
# ---------------------------------------------------------------------------

def load_history():
    if not RESULTS_FILE.exists():
        return []

    history = []

    for line in RESULTS_FILE.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return history


def save_result(result):
    with RESULTS_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


def format_history(history):
    if not history:
        return "No previous experiments."

    lines = []

    for item in history[-20:]:
        status = item.get("status", "?")
        metric = item.get("metric", "?")
        hypothesis = item.get("hypothesis", "")

        lines.append(
            f"Experiment {item.get('experiment')}: "
            f"{status}, "
            f"{METRIC_NAME}={metric}, "
            f"hypothesis={hypothesis}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an autonomous machine-learning research engineer.

Your job is to improve train.py experimentally.

You have:
- program.md: the research instructions
- train.py: the current training program
- experiment history
- the output from the latest training run

You must make ONE coherent improvement per experiment.

Do not merely explain what should change.
Actually return the complete new contents of train.py.

Rules:

1. Preserve the required training/evaluation interface.
2. Do not modify prepare.py.
3. Do not modify the dataset unless program.md explicitly permits it.
4. Do not fake experimental results.
5. Do not claim an improvement until train.py has actually run.
6. Prefer changes supported by a concrete hypothesis.
7. Avoid repeating failed experiments.
8. Keep changes reasonably small so their effect can be understood.
9. Make sure train.py remains executable.
10. Return ONLY JSON matching the requested schema.

Your response must have this exact conceptual structure:

{
  "hypothesis": "short explanation",
  "reasoning": "why this might improve the metric",
  "train_py": "COMPLETE CONTENT OF train.py"
}
"""


def make_prompt(program, train, history):
    return f"""
Here is the research specification:

--- program.md ---
{program}
--- end program.md ---

Here is the current train.py:

--- train.py ---
{train}
--- end train.py ---

Here is recent experiment history:

--- history ---
{format_history(history)}
--- end history ---

Generate the next experiment.

Return JSON with exactly these fields:

{{
  "hypothesis": "...",
  "reasoning": "...",
  "train_py": "..."
}}

The train_py field MUST contain the entire executable train.py,
not a diff and not a code block.
"""


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def validate_candidate(candidate):
    if not isinstance(candidate, dict):
        return False, "Candidate is not a JSON object."

    required = [
        "hypothesis",
        "reasoning",
        "train_py",
    ]

    for field in required:
        if field not in candidate:
            return False, f"Missing field: {field}"

    code = candidate["train_py"]

    if not isinstance(code, str):
        return False, "train_py is not a string."

    if len(code.strip()) < 100:
        return False, "Generated train.py is suspiciously short."

    # Basic protection against accidentally replacing the program
    # with prose.
    if "import " not in code and "from " not in code:
        return False, "Generated train.py does not look like Python."

    return True, ""


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("OLLAMA AUTORESEARCH")
    print("=" * 70)

    print(f"Model:       {OLLAMA_MODEL}")
    print(f"Ollama:      {OLLAMA_URL}")
    print(f"Train:       {TRAIN_COMMAND}")
    print(f"Metric:      {METRIC_NAME}")
    print(f"Direction:   {METRIC_MODE}")
    print(f"Experiments: {MAX_EXPERIMENTS}")
    print()

    if not PROGRAM_FILE.exists():
        die("program.md does not exist.")

    if not TRAIN_FILE.exists():
        die("train.py does not exist.")

    if not git_available():
        die(
            "This script requires a git repository so "
            "experiments can be safely reverted."
        )

    program = read_text(PROGRAM_FILE)

    history = load_history()

    best_metric = None

    for item in history:
        metric = item.get("metric")

        if metric is None:
            continue

        if best_metric is None or is_better(
            metric,
            best_metric,
        ):
            best_metric = metric

    print(f"Current best {METRIC_NAME}: {best_metric}")

    for experiment in range(
        len(history) + 1,
        MAX_EXPERIMENTS + 1,
    ):

        print("\n")
        print("=" * 70)
        print(f"EXPERIMENT {experiment}")
        print("=" * 70)

        train = read_text(TRAIN_FILE)

        prompt = make_prompt(
            program,
            train,
            history,
        )

        print("\nAsking Ollama for the next experiment...")

        candidate = ollama_chat([
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ])

        valid, error = validate_candidate(candidate)

        if not valid:
            print(f"Invalid candidate: {error}")
            continue

        hypothesis = candidate["hypothesis"]
        reasoning = candidate["reasoning"]
        new_train = candidate["train_py"]

        print("\nHypothesis:")
        print(hypothesis)

        print("\nReasoning:")
        print(reasoning)

        # Save the current state before modifying it.
        original_train = train

        # Check syntax before touching the working experiment.
        temporary_path = Path("train_autoresearch_candidate.py")

        write_text(
            temporary_path,
            new_train,
        )

        code, output = run_command(
            f"{sys.executable} -m py_compile "
            f"{temporary_path}",
            timeout=60,
        )

        temporary_path.unlink(
            missing_ok=True
        )

        if code != 0:
            print(
                "\nCandidate failed syntax check. "
                "Skipping experiment."
            )

            save_result({
                "experiment": experiment,
                "status": "invalid",
                "metric": None,
                "hypothesis": hypothesis,
                "reasoning": reasoning,
            })

            history.append({
                "experiment": experiment,
                "status": "invalid",
                "metric": None,
                "hypothesis": hypothesis,
            })

            continue

        # Write candidate.
        write_text(
            TRAIN_FILE,
            new_train,
        )

        # Run the experiment.
        print("\nRunning training experiment...")

        start = time.time()

        code, output = run_command(
            TRAIN_COMMAND,
            timeout=EXPERIMENT_TIMEOUT,
        )

        elapsed = time.time() - start

        metric = extract_metric(output)

        print("\nExperiment finished.")
        print(f"Exit code: {code}")
        print(f"Time:      {elapsed:.1f}s")
        print(f"Metric:    {metric}")

        # Failed training.
        if code != 0 or metric is None:

            print(
                "\nExperiment FAILED. "
                "Restoring previous train.py."
            )

            write_text(
                TRAIN_FILE,
                original_train,
            )

            save_result({
                "experiment": experiment,
                "status": "failed",
                "metric": None,
                "hypothesis": hypothesis,
                "reasoning": reasoning,
                "elapsed": elapsed,
            })

            history.append({
                "experiment": experiment,
                "status": "failed",
                "metric": None,
                "hypothesis": hypothesis,
            })

            continue

        # Decide whether to keep.
        better = is_better(
            metric,
            best_metric,
        )

        if better:

            print(
                f"\nKEEP: {metric} "
                f"is better than {best_metric}"
            )

            best_metric = metric

            # Commit successful experiment.
            if not git_commit_experiment(
                experiment,
                metric,
            ):
                print(
                    "WARNING: Git commit failed. "
                    "Keeping the working tree anyway."
                )

            status = "kept"

        else:

            print(
                f"\nREVERT: {metric} "
                f"is not better than {best_metric}"
            )

            # Restore exact previous file contents.
            write_text(
                TRAIN_FILE,
                original_train,
            )

            status = "reverted"

        result = {
            "experiment": experiment,
            "status": status,
            "metric": metric,
            "best_metric": best_metric,
            "hypothesis": hypothesis,
            "reasoning": reasoning,
            "elapsed": elapsed,
        }

        save_result(result)

        history.append(result)

        print(
            f"\nBest {METRIC_NAME}: "
            f"{best_metric}"
        )

        print(
            "\nWaiting for next experiment..."
        )

    print("\n")
    print("=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)
    print(f"Best {METRIC_NAME}: {best_metric}")


if __name__ == "__main__":
    main()

