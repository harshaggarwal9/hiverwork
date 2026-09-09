"""
Quick reproducibility script for the Hiver AI support agent.

Purpose:
- Verify the repository is runnable.
- Reproduce the headline baseline numbers from committed artifacts.
- Run the final agent on the 200-case golden set.
- Print a compact results summary suitable for the <15-minute requirement.

This script intentionally reuses the project's existing scripts/artifacts
instead of rebuilding the full dataset pipeline from raw Kaggle data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run_command(description: str, command: list[str]) -> None:
    print(f"\n{'=' * 70}")
    print(description)
    print(f"{'=' * 70}")
    print("$", " ".join(command))

    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {description}"
        )


def main() -> None:
    print("Hiver AI Support Agent - Quick Reproduction")
    print(f"Repository: {ROOT}")

    # 1. Verify committed baseline metrics.
    metrics_path = ROOT / "evaluation" / "final_metrics_summary.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing committed metrics artifact: {metrics_path}"
        )

    metrics = pd.read_csv(metrics_path)

    print("\nHeadline results from committed evaluation artifacts:")
    print(metrics.to_string(index=False))

    # 2. Run the final agent on the 200-case golden set.
    run_command(
        "Running final context-aware agent on golden evaluation set",
        [
            sys.executable,
            "scripts/20_update_agent_context.py",
        ],
    )

    # 3. Verify the generated output exists.
    output_path = ROOT / "evaluation" / "context_agent_results.csv"

    if not output_path.exists():
        raise FileNotFoundError(
            f"Expected agent output was not created: {output_path}"
        )

    results = pd.read_csv(output_path)

    print("\nFinal agent output:")
    print(f"Cases evaluated: {len(results)}")

    if "predicted_intent" in results.columns and "intent" in results.columns:
        intent_accuracy = (
            results["predicted_intent"] == results["intent"]
        ).mean()
        print(f"Intent accuracy: {intent_accuracy:.4f}")

    if "predicted_action" in results.columns and "expected_action" in results.columns:
        action_accuracy = (
            results["predicted_action"] == results["expected_action"]
        ).mean()
        print(f"Action accuracy: {action_accuracy:.4f}")

    if "predicted_action" in results.columns:
        print("\nPredicted action distribution:")
        print(results["predicted_action"].value_counts().to_string())

    print("\nQuick reproduction completed successfully.")
    print("See evaluation/final_metrics_summary.csv for committed headline metrics.")


if __name__ == "__main__":
    main()