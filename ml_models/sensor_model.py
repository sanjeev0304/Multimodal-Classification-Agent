"""
ml_models/sensor_model.py
--------------------------
Unsupervised sensor type classifier — matches the model built in the
original sensor clustering project.

Four sensor types identified by value range:
  Temperature  ~15–35°C
  Humidity     ~55–80%
  Moisture     ~400–500 ADC
  Vibration    ~800–900 ADC

Algorithm: GMM (primary) + KMeans (secondary), Hungarian-algorithm
label mapping at training time.

Usage:
    from ml_models.sensor_model import train_sensor_model, predict_sensor
    pipeline = train_sensor_model()          # train + save
    result   = predict_sensor(446.0)         # predict single value
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

MODEL_PATH = os.path.join(os.path.dirname(__file__), "sensor_model.joblib")

SENSOR_PARAMS = {
    "Temperature": {"mean": 25.0,  "std": 3.0},
    "Humidity":    {"mean": 65.0,  "std": 5.0},
    "Moisture":    {"mean": 450.0, "std": 20.0},
    "Vibration":   {"mean": 850.0, "std": 25.0},
}
SENSOR_NAMES = list(SENSOR_PARAMS.keys())
N_SENSORS    = len(SENSOR_NAMES)


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------
def generate_sensor_dataset(n_per_sensor: int = 2500, random_state: int = 42):
    rng = np.random.RandomState(random_state)
    rows = []
    for sensor, params in SENSOR_PARAMS.items():
        values = rng.normal(params["mean"], params["std"], n_per_sensor)
        for v in values:
            rows.append({"value": round(float(v), 2), "ground_truth": sensor})
    df = pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Hungarian mapping helper
# ---------------------------------------------------------------------------
def _hungarian_map(labels_true, labels_pred, n_classes):
    """Map cluster IDs to sensor names using the Hungarian algorithm."""
    cost = np.zeros((n_classes, n_classes), dtype=int)
    for true_idx, name in enumerate(SENSOR_NAMES):
        for cluster_id in range(n_classes):
            cost[true_idx, cluster_id] = np.sum(
                (np.array(labels_true) == name) & (np.array(labels_pred) == cluster_id)
            )
    row_ind, col_ind = linear_sum_assignment(-cost)
    return {col_ind[i]: SENSOR_NAMES[row_ind[i]] for i in range(n_classes)}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_sensor_model(save: bool = True):
    df = generate_sensor_dataset()
    X  = df[["value"]].values

    gmm    = GaussianMixture(n_components=N_SENSORS, random_state=42, n_init=5)
    kmeans = KMeans(n_clusters=N_SENSORS, random_state=42, n_init=10)

    gmm_clusters    = gmm.fit_predict(X)
    kmeans_clusters = kmeans.fit_predict(X)

    gmm_map    = _hungarian_map(df["ground_truth"].tolist(), gmm_clusters.tolist(), N_SENSORS)
    kmeans_map = _hungarian_map(df["ground_truth"].tolist(), kmeans_clusters.tolist(), N_SENSORS)

    # Accuracy on training set
    gmm_preds    = [gmm_map[c]    for c in gmm_clusters]
    kmeans_preds = [kmeans_map[c] for c in kmeans_clusters]
    gmm_acc    = np.mean(np.array(gmm_preds)    == df["ground_truth"].values)
    kmeans_acc = np.mean(np.array(kmeans_preds) == df["ground_truth"].values)

    artifact = {
        "gmm": gmm, "kmeans": kmeans,
        "gmm_map": gmm_map, "kmeans_map": kmeans_map,
        "gmm_train_acc": float(gmm_acc),
        "kmeans_train_acc": float(kmeans_acc),
    }
    if save:
        joblib.dump(artifact, MODEL_PATH)
    return artifact


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
def predict_sensor(value: float, artifact=None):
    """
    Predict sensor type for a single raw value.
    Returns dict with gmm_prediction, kmeans_prediction, confidence_score.
    """
    if artifact is None:
        if not os.path.exists(MODEL_PATH):
            artifact = train_sensor_model()
        else:
            artifact = joblib.load(MODEL_PATH)

    X = np.array([[value]])

    gmm_cluster    = int(artifact["gmm"].predict(X)[0])
    kmeans_cluster = int(artifact["kmeans"].predict(X)[0])
    gmm_proba      = artifact["gmm"].predict_proba(X)[0]

    gmm_pred    = artifact["gmm_map"][gmm_cluster]
    kmeans_pred = artifact["kmeans_map"][kmeans_cluster]

    # Confidence = max GMM posterior (calibrated probability)
    mapped_proba = {artifact["gmm_map"][i]: float(p) for i, p in enumerate(gmm_proba)}
    confidence   = mapped_proba.get(gmm_pred, 0.0)

    return {
        "gmm_prediction":    gmm_pred,
        "kmeans_prediction": kmeans_pred,
        "confidence_score":  round(confidence, 4),
        "all_probabilities": {k: round(v, 4) for k, v in mapped_proba.items()},
    }


# ---------------------------------------------------------------------------
# Generate test CSV
# ---------------------------------------------------------------------------
def generate_test_csv(n_per_sensor: int = 8, path: str = "test_data/sensor_test.csv"):
    rng = np.random.RandomState(99)
    rows = []
    for sensor, params in SENSOR_PARAMS.items():
        values = rng.normal(params["mean"], params["std"], n_per_sensor)
        for v in values:
            rows.append({"value": round(float(v), 2), "ground_truth": sensor})
    df = pd.DataFrame(rows).sample(frac=1, random_state=99).reset_index(drop=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    print("Training sensor model...")
    art = train_sensor_model()
    print(f"  GMM train accuracy:    {art['gmm_train_acc']:.1%}")
    print(f"  KMeans train accuracy: {art['kmeans_train_acc']:.1%}")
    r = predict_sensor(25.3)
    print(f"  Sample prediction (25.3): {r['gmm_prediction']} ({r['confidence_score']:.1%})")
