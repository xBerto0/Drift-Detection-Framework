"""Esperimento ADWIN su dataset reale ELEC2.

Testa la strategia ADWIN su tre tipi di drift:
- Feature drift: sulle 8 feature del dataset
- Prediction drift: sullo stream delle predizioni del classificatore
- Concept drift: sullo stream degli errori del classificatore
  (setup fedele al paper Bifet & Gavalda 2006, sezione 5.3, "ADWIN esterno")
"""

from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from config import (
    ADWIN_DELTA,
    DATASET_ELEC2_PATH,
    ELEC2_TRAIN_SIZE,
    FEATURE_K_THRESHOLD,
)
from detectors.adwin_strategy import ADWINStrategy
from detectors.concept_drift_detector import ConceptDriftDetector
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.prediction_drift_detector import PredictionDriftDetector


# --- Parametri dell'esperimento (caricati da .env tramite config.py) ---
DATASET_PATH = DATASET_ELEC2_PATH
TRAIN_SIZE = ELEC2_TRAIN_SIZE
DELTA = ADWIN_DELTA

# --- Caricamento del dataset ---
df = pd.read_csv(DATASET_PATH)
feature_names = [c for c in df.columns if c != "class"]
X = df[feature_names].values.astype(float)
y_raw = df["class"].values
y = np.array([1 if v == "UP" else 0 for v in y_raw])

n_samples, n_features = X.shape

# --- Training del classificatore ---
X_train = X[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]
classifier = GaussianNB()
classifier.fit(X_train, y_train)

train_accuracy = classifier.score(X_train, y_train)

# --- Preparazione dei detector ---
feature_detector = FeatureDriftDetector(
    strategy_cls=ADWINStrategy,
    n_features=n_features,
    k=FEATURE_K_THRESHOLD,
    feature_names=feature_names,
    delta=DELTA,
)
prediction_detector = PredictionDriftDetector(
    strategy_cls=ADWINStrategy,
    delta=DELTA,
)
concept_detector = ConceptDriftDetector(
    strategy_cls=ADWINStrategy,
    delta=DELTA,
)

# --- Preparazione dei file di output ---
output_dir = Path("results/ADWIN")
output_dir.mkdir(parents=True, exist_ok=True)

feature_output_path = output_dir / "ADWIN_feature_elec2.txt"
prediction_output_path = output_dir / "ADWIN_prediction_elec2.txt"
concept_output_path = output_dir / "ADWIN_concept_elec2.txt"

# --- Streaming ---
feature_drift_steps = []
feature_drift_counter = Counter()
prediction_drift_steps = []
concept_drift_steps = []
rolling_correct = deque(maxlen=1000)
accuracy_snapshots = []

for t in range(TRAIN_SIZE, n_samples):
    x_t = X[t]
    y_true_t = y[t]
    y_pred_t = classifier.predict([x_t])[0]

    rolling_correct.append(1 if y_pred_t == y_true_t else 0)
    if (t - TRAIN_SIZE + 1) % 1000 == 0:
        acc = sum(rolling_correct) / len(rolling_correct)
        accuracy_snapshots.append((t, acc))

    # Feature drift
    feature_detector.update(x_t)
    feat_result = feature_detector.detect()
    if feat_result.drift_detected:
        feature_drift_steps.append(t)
        for f in feat_result.metadata["drifted_features"]:
            feature_drift_counter[f] += 1

    # Prediction drift
    prediction_detector.update(y_pred_t)
    pred_result = prediction_detector.detect()
    if pred_result.drift_detected:
        prediction_drift_steps.append(t)

    # Concept drift (il detector calcola internamente l'errore)
    concept_detector.update(y_pred_t, y_true_t)
    conc_result = concept_detector.detect()
    if conc_result.drift_detected:
        concept_drift_steps.append(t)


# --- Report FEATURE DRIFT ---
with open(feature_output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: ADWIN Feature Drift su ELEC2")
    log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log()
    log("Configurazione:")
    log(f"  Dataset: {DATASET_PATH}")
    log(f"  Campioni totali: {n_samples}")
    log(f"  Feature ({n_features}): {feature_names}")
    log(f"  Campioni di training: {TRAIN_SIZE}")
    log(f"  Campioni in streaming: {n_samples - TRAIN_SIZE}")
    log(f"  Classificatore: GaussianNB (statico)")
    log(f"  Accuracy sul training set: {train_accuracy:.4f}")
    log(f"  Detector: FeatureDriftDetector(ADWINStrategy)")
    log(f"  Parametri: delta={DELTA}, k=1")
    log()
    log("-" * 70)
    log("Risultati feature drift")
    log("-" * 70)
    log(f"Segnalazioni di drift globale: {len(feature_drift_steps)}")
    if feature_drift_steps:
        log(f"Primo drift al passo t = {feature_drift_steps[0]}")
        log(f"Ultimo drift al passo t = {feature_drift_steps[-1]}")
    log()
    log("Frequenza di drift per feature:")
    for name in feature_names:
        log(f"  {name}: {feature_drift_counter[name]} volte")
    log()
    log("Accuracy del classificatore (media mobile su 1000 campioni):")
    for step, acc in accuracy_snapshots:
        log(f"  passo {step}: {acc:.4f}")


# --- Report PREDICTION DRIFT ---
with open(prediction_output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: ADWIN Prediction Drift su ELEC2")
    log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log()
    log("Configurazione:")
    log(f"  Dataset: {DATASET_PATH}")
    log(f"  Campioni totali: {n_samples}")
    log(f"  Campioni di training: {TRAIN_SIZE}")
    log(f"  Campioni in streaming: {n_samples - TRAIN_SIZE}")
    log(f"  Classificatore: GaussianNB (statico)")
    log(f"  Detector: PredictionDriftDetector(ADWINStrategy)")
    log(f"  Stream monitorato: y_pred (binario UP=1, DOWN=0)")
    log(f"  Parametri: delta={DELTA}")
    log()
    log("-" * 70)
    log("Risultati prediction drift")
    log("-" * 70)
    log(f"Segnalazioni di drift totali: {len(prediction_drift_steps)}")
    if prediction_drift_steps:
        log(f"Primo drift al passo t = {prediction_drift_steps[0]}")
        log(f"Ultimo drift al passo t = {prediction_drift_steps[-1]}")
    log()
    log("Elenco delle segnalazioni:")
    for step in prediction_drift_steps:
        log(f"  passo t = {step}")


# --- Report CONCEPT DRIFT ---
with open(concept_output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: ADWIN Concept Drift su ELEC2")
    log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log()
    log("Configurazione:")
    log(f"  Dataset: {DATASET_PATH}")
    log(f"  Campioni totali: {n_samples}")
    log(f"  Campioni di training: {TRAIN_SIZE}")
    log(f"  Campioni in streaming: {n_samples - TRAIN_SIZE}")
    log(f"  Classificatore: GaussianNB (statico)")
    log(f"  Accuracy sul training set: {train_accuracy:.4f}")
    log(f"  Detector: ConceptDriftDetector(ADWINStrategy)")
    log(f"  Stream monitorato: errore binario (1 se y_pred != y_true)")
    log(f"  Parametri: delta={DELTA}")
    log()
    log("Nota: setup analogo alla configurazione 'ADWIN esterno' del paper")
    log("      Bifet & Gavalda 2006, sezione 5.1.")
    log()
    log("-" * 70)
    log("Risultati concept drift")
    log("-" * 70)
    log(f"Segnalazioni di drift totali: {len(concept_drift_steps)}")
    if concept_drift_steps:
        log(f"Primo drift al passo t = {concept_drift_steps[0]}")
        log(f"Ultimo drift al passo t = {concept_drift_steps[-1]}")
    log()
    log("Elenco delle segnalazioni:")
    for step in concept_drift_steps:
        log(f"  passo t = {step}")
    log()
    log("Accuracy del classificatore (media mobile su 1000 campioni):")
    log("Utile per confrontare i punti in cui l'accuracy cala con")
    log("i punti in cui ADWIN ha segnalato drift.")
    for step, acc in accuracy_snapshots:
        log(f"  passo {step}: {acc:.4f}")


print()
print(f"Output salvati in:")
print(f"  {feature_output_path}")
print(f"  {prediction_output_path}")
print(f"  {concept_output_path}")
