"""Esperimento KS su dataset reale ELEC2.

Testa la strategia Kolmogorov-Smirnov su due tipi di drift:
- Feature drift: sulle 8 feature del dataset (naturale per KS, continue)
- Prediction drift: sullo stream delle predizioni del classificatore
  (incluso come confronto: KS su stream binari e' subottimale)

Il concept drift non viene testato con KS: e' piu' appropriato ADWIN.
"""

import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.ks_strategy import KSStrategy
from detectors.prediction_drift_detector import PredictionDriftDetector

# Silenzia i warning di scipy sul fallback all'approssimazione asintotica del KS
# quando applicato a dati binari (comportamento atteso, non un errore).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


# --- Parametri dell'esperimento ---
DATASET_PATH = "data/electricity-normalized.csv"
TRAIN_SIZE = 500
WINDOW_SIZE = 200
ALPHA = 0.05

# --- Caricamento del dataset ---
df = pd.read_csv(DATASET_PATH)
feature_names = [c for c in df.columns if c != "class"]
X = df[feature_names].values.astype(float)
y_raw = df["class"].values
# Conversione UP/DOWN in 1/0
y = np.array([1 if v == "UP" else 0 for v in y_raw])

n_samples, n_features = X.shape

# --- Training del classificatore ---
X_train = X[:TRAIN_SIZE]
y_train = y[:TRAIN_SIZE]
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Accuracy iniziale sul training set (informativa)
train_accuracy = classifier.score(X_train, y_train)

# --- Preparazione dei detector ---
feature_detector = FeatureDriftDetector(
    strategy_cls=KSStrategy,
    n_features=n_features,
    k=1,
    feature_names=feature_names,
    window_size=WINDOW_SIZE,
    alpha=ALPHA,
)
prediction_detector = PredictionDriftDetector(
    strategy_cls=KSStrategy,
    window_size=WINDOW_SIZE,
    alpha=ALPHA,
)

# --- Preparazione dei file di output ---
output_dir = Path("results/KS")
output_dir.mkdir(parents=True, exist_ok=True)

feature_output_path = output_dir / "KS_feature_elec2.txt"
prediction_output_path = output_dir / "KS_prediction_elec2.txt"

# --- Streaming ---
feature_drift_steps = []
feature_drift_counter = Counter()
prediction_drift_steps = []
rolling_correct = deque(maxlen=1000)
accuracy_snapshots = []  # (step, accuracy)

for t in range(TRAIN_SIZE, n_samples):
    x_t = X[t]
    y_true_t = y[t]
    y_pred_t = classifier.predict([x_t])[0]

    # Aggiornamento accuracy mobile
    rolling_correct.append(1 if y_pred_t == y_true_t else 0)

    # Snapshot dell'accuracy ogni 1000 passi
    if (t - TRAIN_SIZE + 1) % 1000 == 0:
        acc = sum(rolling_correct) / len(rolling_correct)
        accuracy_snapshots.append((t, acc))

    # Feature drift detector: riceve il vettore intero
    feature_detector.update(x_t)
    feat_result = feature_detector.detect()
    if feat_result.drift_detected:
        feature_drift_steps.append(t)
        for f in feat_result.metadata["drifted_features"]:
            feature_drift_counter[f] += 1

    # Prediction drift detector: riceve la sola predizione
    prediction_detector.update(y_pred_t)
    pred_result = prediction_detector.detect()
    if pred_result.drift_detected:
        prediction_drift_steps.append(t)


# --- Report finale su FEATURE DRIFT ---
with open(feature_output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: KS Feature Drift su ELEC2")
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
    log(f"  Detector: FeatureDriftDetector(KSStrategy)")
    log(f"  Parametri: window_size={WINDOW_SIZE}, alpha={ALPHA}, k=1")
    log()
    log("-" * 70)
    log("Risultati feature drift")
    log("-" * 70)
    log(f"Segnalazioni di drift globale (>=k feature drittate): {len(feature_drift_steps)}")
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


# --- Report finale su PREDICTION DRIFT ---
with open(prediction_output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: KS Prediction Drift su ELEC2")
    log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log()
    log("Configurazione:")
    log(f"  Dataset: {DATASET_PATH}")
    log(f"  Campioni totali: {n_samples}")
    log(f"  Campioni di training: {TRAIN_SIZE}")
    log(f"  Campioni in streaming: {n_samples - TRAIN_SIZE}")
    log(f"  Classificatore: GaussianNB (statico)")
    log(f"  Detector: PredictionDriftDetector(KSStrategy)")
    log(f"  Stream monitorato: y_pred (binario UP=1, DOWN=0)")
    log(f"  Parametri: window_size={WINDOW_SIZE}, alpha={ALPHA}")
    log()
    log("Nota: KS su stream binari ha potere statistico ridotto.")
    log("      Include come confronto con ADWIN, che gestisce nativamente")
    log("      dati discreti.")
    log()
    log("-" * 70)
    log("Risultati prediction drift")
    log("-" * 70)
    log(f"Segnalazioni di drift totali: {len(prediction_drift_steps)}")
    if prediction_drift_steps:
        log(f"Primo drift al passo t = {prediction_drift_steps[0]}")
        log(f"Ultimo drift al passo t = {prediction_drift_steps[-1]}")
    log()
    log(f"Elenco delle prime 20 segnalazioni:")
    for step in prediction_drift_steps[:20]:
        log(f"  passo t = {step}")


print()
print(f"Output salvati in:")
print(f"  {feature_output_path}")
print(f"  {prediction_output_path}")
