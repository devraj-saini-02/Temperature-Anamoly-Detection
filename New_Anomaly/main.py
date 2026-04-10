"""Generate a labeled anomaly dataset and train a simple ML model.

This script creates a CSV dataset where each row contains:
  - vector: a list of 5 temperature readings (as a JSON string)
  - label: 0 for normal, 1 for anomaly

The "normal" sequences are centered around 356C with small noise.
Anomalies contain a sudden spike (a large jump) in temperature.

Usage:
  python main.py --output dataset.csv --samples 5000 --anomaly-rate 0.15
  python main.py --train --input dataset.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def generate_sequence(is_anomaly: bool, length: int = 5) -> list[float]:
    """Generate a temperature sequence.

    Args:
        is_anomaly: If True, inserts a sudden spike in one of the timestamps.
        length: Number of timestamps in the vector (default 5).

    Returns:
        List of float temperatures.
    """

    base_temp = 35.0
    noise_std = 0.4

    # Normal noise around the base temperature
    seq = list(np.random.normal(loc=base_temp, scale=noise_std, size=length))

    if is_anomaly:
        # Choose a position for the spike (not always the first or last to make it more realistic)
        spike_idx = random.randrange(length)

        # Add a sharp jump to simulate a sudden temperature rise
        spike_amount = random.uniform(15.0, 30.0)
        seq[spike_idx] += spike_amount

        # Optionally bump nearby readings slightly so the pattern is more "spike-like"
        for neigh in (spike_idx - 1, spike_idx + 1):
            if 0 <= neigh < length:
                seq[neigh] += random.uniform(1.0, 3.0)

    return [float(x) for x in seq]


def build_dataset(
    num_samples: int = 1000,
    anomaly_rate: float = 0.1,
    length: int = 5,
) -> pd.DataFrame:
    """Create a labeled dataset of temperature vectors.

    The returned DataFrame has two columns:
      - vector: JSON-encoded list of floats
      - label: 0 (normal) or 1 (anomaly)
    """

    rows: list[dict[str, object]] = []
    for _ in range(num_samples):
        is_anomaly = random.random() < anomaly_rate
        seq = generate_sequence(is_anomaly, length=length)
        rows.append({"vector": json.dumps(seq), "label": int(is_anomaly)})

    return pd.DataFrame(rows)


def vector_to_features(vec: list[float]) -> dict[str, float]:
    """Extract features from a temperature vector for model training."""

    arr = np.asarray(vec, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "range": float(arr.max() - arr.min()),
        "diff_1": float(arr[1] - arr[0]) if arr.size >= 2 else 0.0,
        "diff_2": float(arr[2] - arr[1]) if arr.size >= 3 else 0.0,
        "diff_3": float(arr[3] - arr[2]) if arr.size >= 4 else 0.0,
        "diff_4": float(arr[4] - arr[3]) if arr.size >= 5 else 0.0,
    }


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the CSV dataset into a feature matrix and label vector."""

    features = []
    for s in df["vector"]:
        vec = json.loads(s)
        features.append(vector_to_features(vec))

    X = pd.DataFrame(features)
    y = df["label"].astype(int)
    return X, y


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> None:
    """Train a RandomForest model and print evaluation metrics."""

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Model evaluation ===")
    print(f"Samples: {len(df)} (train={len(X_train)}, test={len(X_test)})")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a temperature anomaly dataset and optionally train a model."
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV path to save the generated dataset.",
        default="dataset.csv",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=5000,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--anomaly-rate",
        "-a",
        type=float,
        default=0.12,
        help="Fraction of samples that should be anomalies.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a model using the generated (or existing) dataset.",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to an existing dataset CSV to train on (if not specified, uses --output).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducible dataset generation.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_path = Path(args.output)

    if args.train and args.input:
        dataset_path = Path(args.input)
    else:
        dataset_path = output_path

    if not dataset_path.exists() or dataset_path == output_path:
        df = build_dataset(
            num_samples=args.samples, anomaly_rate=args.anomaly_rate, length=5
        )
        df.to_csv(dataset_path, index=False)
        print(f"Saved dataset: {dataset_path} (samples={len(df)})")
    else:
        df = pd.read_csv(dataset_path)
        print(f"Loaded existing dataset: {dataset_path} (samples={len(df)})")

    if args.train:
        train_and_evaluate(df)


if __name__ == "__main__":
    main()
