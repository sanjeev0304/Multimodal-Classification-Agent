"""
ml_models/train_all.py
-----------------------
Train both ML models, export training data to CSV, and generate test CSVs.

Run from the project root (same folder as streamlit_app.py):
    python ml_models/train_all.py

Output files:
    training_data/sensor_training_data.csv   — 10,000 rows used to train the sensor model
    training_data/network_training_data.csv  — 10,002 rows used to train the network model
    test_data/sensor_test.csv                — 32 labeled rows for accuracy testing
    test_data/network_test.csv               — 24 labeled rows for accuracy testing
    ml_models/sensor_model.joblib            — trained sensor model (GMM + KMeans)
    ml_models/network_model.joblib           — trained network model (Random Forest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.sensor_model  import train_sensor_model,  generate_sensor_dataset, generate_test_csv as sensor_test_csv
from ml_models.network_model import train_network_model, generate_network_dataset, generate_test_csv as network_test_csv


def main():
    os.makedirs("training_data", exist_ok=True)
    os.makedirs("test_data",     exist_ok=True)

    # ── Sensor model ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("Training Sensor Model (GMM + KMeans)")
    print("=" * 55)
    art_s = train_sensor_model()
    print(f"  GMM    train accuracy : {art_s['gmm_train_acc']:.1%}")
    print(f"  KMeans train accuracy : {art_s['kmeans_train_acc']:.1%}")
    print(f"  Saved  → ml_models/sensor_model.joblib")

    # Export full training data
    sensor_train_df = generate_sensor_dataset(n_per_sensor=2500)
    sensor_train_df.to_csv("training_data/sensor_training_data.csv", index=False)
    print(f"  Training data → training_data/sensor_training_data.csv  ({len(sensor_train_df):,} rows)")
    print(f"  Class distribution:")
    for cls, cnt in sensor_train_df["ground_truth"].value_counts().items():
        print(f"    {cls:12}  {cnt:,} rows")

    # ── Network model ─────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("Training Network Model (Random Forest)")
    print("=" * 55)
    art_n = train_network_model()
    print(f"  Train accuracy : {art_n['train_acc']:.1%}")
    print(f"  Saved  → ml_models/network_model.joblib")

    # Export full training data
    network_train_df = generate_network_dataset(n_per_class=3334)
    network_train_df.to_csv("training_data/network_training_data.csv", index=False)
    print(f"  Training data → training_data/network_training_data.csv  ({len(network_train_df):,} rows)")
    print(f"  Class distribution:")
    for cls, cnt in network_train_df["ground_truth"].value_counts().items():
        print(f"    {cls:12}  {cnt:,} rows")

    # ── Test CSVs ─────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("Generating Test CSVs (with ground_truth labels)")
    print("=" * 55)
    print()
    print("  What is ground_truth?")
    print("  ─────────────────────────────────────────────────────")
    print("  ground_truth is the CORRECT ANSWER we already know")
    print("  for each test row. It lets us measure how accurate")
    print("  the model is — we compare the model's prediction")
    print("  against ground_truth and count how many match.")
    print()
    print("  Think of it like an answer key for an exam:")
    print("    Row input  → model predicts → we check vs ground_truth")
    print("    25.3       → Temperature    → ground_truth=Temperature ✓")
    print("    65.1       → Temperature    → ground_truth=Humidity    ✗")
    print()
    print("  The test data is SEPARATE from training data — the model")
    print("  never sees these rows during training, so the accuracy")
    print("  it scores here is an honest measure of real performance.")
    print("  ─────────────────────────────────────────────────────")
    print()

    s_df = sensor_test_csv(n_per_sensor=8,  path="test_data/sensor_test.csv")
    n_df = network_test_csv(n_per_class=8,  path="test_data/network_test.csv")
    print(f"  sensor_test.csv  → {len(s_df)} rows  (test_data/)")
    print(f"  network_test.csv → {len(n_df)} rows  (test_data/)")

    print()
    print("=" * 55)
    print("Summary of all generated files")
    print("=" * 55)
    files = [
        ("ml_models/sensor_model.joblib",                 "Trained GMM+KMeans — what the app uses for predictions"),
        ("ml_models/network_model.joblib",                "Trained Random Forest — what the app uses for predictions"),
        ("training_data/sensor_training_data.csv",        "10,000 rows the sensor model was trained on"),
        ("training_data/network_training_data.csv",       "10,002 rows the network model was trained on"),
        ("test_data/sensor_test.csv",                     "32 labeled rows to measure sensor model accuracy"),
        ("test_data/network_test.csv",                    "24 labeled rows to measure network model accuracy"),
    ]
    for path, desc in files:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {path}")
        print(f"      {desc}")
    print()
    print("Done. Run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()