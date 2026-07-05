"""Test ADWIN + FeatureDriftDetector su feature Bernoulli con drift artificiale.

Stesso scenario del test KS per permettere un confronto diretto:
Figura 2 del paper Bifet & Gavalda 2006, cambio brusco di mu al passo 1000.
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from data.synthetic_generator import bernoulli_with_abrupt_drift
from detectors.adwin_strategy import ADWINStrategy
from detectors.feature_drift_detector import FeatureDriftDetector


# --- Parametri dell'esperimento ---
N_SAMPLES = 2000
DRIFT_POINT = 1000
SEED = 42
DELTA = 0.002

# --- Generazione delle 3 feature (uguale al test KS) ---
feature_0 = bernoulli_with_abrupt_drift(
    n_samples=N_SAMPLES, p_before=0.2, p_after=0.2,
    drift_point=DRIFT_POINT, seed=SEED,
)
feature_1 = bernoulli_with_abrupt_drift(
    n_samples=N_SAMPLES, p_before=0.8, p_after=0.4,
    drift_point=DRIFT_POINT, seed=SEED + 1,
)
feature_2 = bernoulli_with_abrupt_drift(
    n_samples=N_SAMPLES, p_before=0.5, p_after=0.5,
    drift_point=DRIFT_POINT, seed=SEED + 2,
)

data = np.column_stack([feature_0, feature_1, feature_2])

# --- Setup del detector con ADWIN ---
detector = FeatureDriftDetector(
    strategy_cls=ADWINStrategy,
    n_features=3,
    k=1,
    feature_names=["f0_stable", "f1_drift", "f2_stable"],
    delta=DELTA,
)

# --- Streaming ---
# ADWIN segnala drift_detected=True solo nell'istante esatto del taglio,
# quindi tracciamo tutti i passi in cui scatta l'allarme.
drift_steps = []
drift_features_per_step = []

for t in range(N_SAMPLES):
    detector.update(data[t])
    result = detector.detect()
    if result.drift_detected:
        drift_steps.append(t)
        drift_features_per_step.append(result.metadata["drifted_features"])

# Riassunto per feature
feature_counts = {}
for features in drift_features_per_step:
    for f in features:
        feature_counts[f] = feature_counts.get(f, 0) + 1

# --- Output su file ---
output_dir = Path("results/ADWIN")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "ADWIN_feature_bernoulli.txt"

with open(output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: ADWIN Feature Drift su Bernoulli sintetica")
    log(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log()
    log("Configurazione:")
    log(f"  Scenario: Figura 2 di Bifet & Gavalda 2006 (cambio brusco al passo {DRIFT_POINT})")
    log(f"  Numero campioni: {N_SAMPLES}")
    log(f"  Numero feature: 3")
    log(f"  f0_stable: Bernoulli(p=0.2) per tutti i passi")
    log(f"  f1_drift : Bernoulli(p=0.8) fino a t={DRIFT_POINT}, poi Bernoulli(p=0.4)")
    log(f"  f2_stable: Bernoulli(p=0.5) per tutti i passi")
    log(f"  Seed base: {SEED}")
    log()
    log(f"  Detector: FeatureDriftDetector(ADWINStrategy)")
    log(f"  Parametri: delta={DELTA}, k=1")
    log()
    log("-" * 70)
    log("Risultati")
    log("-" * 70)
    log(f"Numero totale di segnali di drift: {len(drift_steps)}")

    if drift_steps:
        log(f"Primo segnale al passo t = {drift_steps[0]}")
        log(f"  Feature drittate: {drift_features_per_step[0]}")
        delay = drift_steps[0] - DRIFT_POINT
        log(f"  Ritardo rispetto al cambio reale (t={DRIFT_POINT}): {delay} passi")
        log()
        log("Segnali per feature:")
        for feature, count in feature_counts.items():
            log(f"  {feature}: {count} volte")
    else:
        log("Nessun drift rilevato durante il flusso.")

print()
print(f"Output salvato in: {output_path}")
