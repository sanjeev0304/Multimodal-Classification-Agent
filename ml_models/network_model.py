"""
ml_models/network_model.py
---------------------------
Supervised network traffic classifier.

Predicts: Normal / Suspicious / Priority
from 6 flow-level features extracted per connection.

Features:
  avg_packet_size   bytes    typical web ~800, scan ~64
  byte_rate         B/s      streaming ~50k, scan ~200
  duration          seconds  long flows = priority/streaming
  tcp_pct           0-1      scans mix UDP/ICMP
  unique_dst_ports  count    scans = many ports
  syn_ratio         0-1      SYN flood = high

Algorithm: Random Forest (supervised, labeled synthetic data)

The original network clustering project classified device type/IP
(unsupervised). This model aligns with the Gemini MCP tool which
classifies Normal/Suspicious/Priority — enabling a direct comparison.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "network_model.joblib")

FEATURES = [
    "avg_packet_size",
    "byte_rate",
    "duration",
    "tcp_pct",
    "unique_dst_ports",
    "syn_ratio",
]
LABELS = ["Normal", "Suspicious", "Priority"]


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------
def generate_network_dataset(n_per_class: int = 1500, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    rows = []

    # ── Normal — typical web/app traffic ────────────────────────────────────
    for _ in range(n_per_class):
        rows.append({
            "avg_packet_size":  rng.uniform(400, 1400),
            "byte_rate":        rng.uniform(1000, 60000),
            "duration":         rng.uniform(0.1, 10.0),
            "tcp_pct":          rng.uniform(0.6, 0.95),
            "unique_dst_ports": rng.randint(1, 6),
            "syn_ratio":        rng.uniform(0.02, 0.15),
            "ground_truth":     "Normal",
        })

    # ── Suspicious — port scans, SYN floods, anomalous patterns ─────────────
    for _ in range(n_per_class):
        pattern = rng.choice(["port_scan", "syn_flood", "small_flood"])
        if pattern == "port_scan":
            rows.append({
                "avg_packet_size":  rng.uniform(40, 120),
                "byte_rate":        rng.uniform(100, 2000),
                "duration":         rng.uniform(0.01, 1.0),
                "tcp_pct":          rng.uniform(0.3, 0.7),
                "unique_dst_ports": rng.randint(20, 200),
                "syn_ratio":        rng.uniform(0.5, 0.9),
                "ground_truth":     "Suspicious",
            })
        elif pattern == "syn_flood":
            rows.append({
                "avg_packet_size":  rng.uniform(40, 80),
                "byte_rate":        rng.uniform(5000, 50000),
                "duration":         rng.uniform(0.5, 5.0),
                "tcp_pct":          rng.uniform(0.85, 1.0),
                "unique_dst_ports": rng.randint(1, 3),
                "syn_ratio":        rng.uniform(0.7, 1.0),
                "ground_truth":     "Suspicious",
            })
        else:
            rows.append({
                "avg_packet_size":  rng.uniform(40, 200),
                "byte_rate":        rng.uniform(10000, 80000),
                "duration":         rng.uniform(0.01, 0.5),
                "tcp_pct":          rng.uniform(0.1, 0.5),
                "unique_dst_ports": rng.randint(10, 100),
                "syn_ratio":        rng.uniform(0.3, 0.7),
                "ground_truth":     "Suspicious",
            })

    # ── Priority — high-volume critical traffic (video, large transfers) ─────
    for _ in range(n_per_class):
        pattern = rng.choice(["streaming", "bulk_transfer", "voip"])
        if pattern == "streaming":
            rows.append({
                "avg_packet_size":  rng.uniform(1000, 1500),
                "byte_rate":        rng.uniform(80000, 500000),
                "duration":         rng.uniform(30, 600),
                "tcp_pct":          rng.uniform(0.5, 0.9),
                "unique_dst_ports": rng.randint(1, 4),
                "syn_ratio":        rng.uniform(0.01, 0.05),
                "ground_truth":     "Priority",
            })
        elif pattern == "bulk_transfer":
            rows.append({
                "avg_packet_size":  rng.uniform(800, 1500),
                "byte_rate":        rng.uniform(100000, 1000000),
                "duration":         rng.uniform(5, 300),
                "tcp_pct":          rng.uniform(0.9, 1.0),
                "unique_dst_ports": rng.randint(1, 3),
                "syn_ratio":        rng.uniform(0.01, 0.04),
                "ground_truth":     "Priority",
            })
        else:  # voip
            rows.append({
                "avg_packet_size":  rng.uniform(100, 300),
                "byte_rate":        rng.uniform(20000, 100000),
                "duration":         rng.uniform(10, 3600),
                "tcp_pct":          rng.uniform(0.1, 0.4),
                "unique_dst_ports": rng.randint(1, 3),
                "syn_ratio":        rng.uniform(0.01, 0.05),
                "ground_truth":     "Priority",
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    # Round to sensible precision
    for col in ["avg_packet_size", "byte_rate", "duration", "tcp_pct", "syn_ratio"]:
        df[col] = df[col].round(2)
    df["unique_dst_ports"] = df["unique_dst_ports"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_network_model(save: bool = True):
    df = generate_network_dataset()
    X  = df[FEATURES]
    y  = df["ground_truth"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X, y)

    train_acc = pipeline.score(X, y)
    artifact  = {"pipeline": pipeline, "train_acc": float(train_acc)}
    if save:
        joblib.dump(artifact, MODEL_PATH)
    return artifact


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict_network(row: dict, artifact=None):
    """
    Predict Normal/Suspicious/Priority for one flow record.
    row must contain keys matching FEATURES (missing keys default to 0).
    """
    if artifact is None:
        if not os.path.exists(MODEL_PATH):
            artifact = train_network_model()
        else:
            artifact = joblib.load(MODEL_PATH)

    pipeline = artifact["pipeline"]
    X        = pd.DataFrame([{f: row.get(f, 0.0) for f in FEATURES}])
    pred     = pipeline.predict(X)[0]
    proba    = pipeline.predict_proba(X)[0]
    classes  = list(pipeline.classes_)
    conf     = float(proba[classes.index(pred)])

    return {
        "classification":  pred,
        "confidence_score": round(conf, 4),
        "all_probabilities": {
            classes[i]: round(float(proba[i]), 4) for i in range(len(classes))
        },
    }


# ---------------------------------------------------------------------------
# Generate test CSV
# ---------------------------------------------------------------------------
def generate_test_csv(n_per_class: int = 8, path: str = "test_data/network_test.csv"):
    rng = np.random.RandomState(77)

    normal_rows = [{
        "avg_packet_size":  round(rng.uniform(400, 1400), 2),
        "byte_rate":        round(rng.uniform(1000, 60000), 2),
        "duration":         round(rng.uniform(0.1, 10.0), 2),
        "tcp_pct":          round(rng.uniform(0.6, 0.95), 2),
        "unique_dst_ports": int(rng.randint(1, 6)),
        "syn_ratio":        round(rng.uniform(0.02, 0.15), 2),
        "ground_truth":     "Normal",
    } for _ in range(n_per_class)]

    susp_rows = [{
        "avg_packet_size":  round(rng.uniform(40, 120), 2),
        "byte_rate":        round(rng.uniform(100, 5000), 2),
        "duration":         round(rng.uniform(0.01, 1.0), 2),
        "tcp_pct":          round(rng.uniform(0.3, 0.7), 2),
        "unique_dst_ports": int(rng.randint(20, 150)),
        "syn_ratio":        round(rng.uniform(0.5, 0.95), 2),
        "ground_truth":     "Suspicious",
    } for _ in range(n_per_class)]

    prio_rows = [{
        "avg_packet_size":  round(rng.uniform(900, 1500), 2),
        "byte_rate":        round(rng.uniform(100000, 500000), 2),
        "duration":         round(rng.uniform(30, 300), 2),
        "tcp_pct":          round(rng.uniform(0.7, 1.0), 2),
        "unique_dst_ports": int(rng.randint(1, 3)),
        "syn_ratio":        round(rng.uniform(0.01, 0.04), 2),
        "ground_truth":     "Priority",
    } for _ in range(n_per_class)]

    df = pd.DataFrame(normal_rows + susp_rows + prio_rows)
    df = df.sample(frac=1, random_state=77).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    print("Training network model...")
    art = train_network_model()
    print(f"  Train accuracy: {art['train_acc']:.1%}")
    r = predict_network({
        "avg_packet_size": 60, "byte_rate": 300, "duration": 0.2,
        "tcp_pct": 0.5, "unique_dst_ports": 80, "syn_ratio": 0.75,
    })
    print(f"  Sample prediction (port scan): {r['classification']} ({r['confidence_score']:.1%})")
