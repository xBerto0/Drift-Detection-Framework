"""Test KS + FeatureDriftDetector su feature Bernoulli con drift artificiale.

Replica lo scenario della Figura 2 del paper Bifet & Gavalda 2006:
cambio brusco di mu al passo 1000.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    BERNOULLI_DRIFT_POINT,
    BERNOULLI_N_SAMPLES,
    BERNOULLI_SEED,
    FEATURE_K_THRESHOLD,
    KS_ALPHA,
    KS_WINDOW_SIZE,
)
from data.synthetic_generator import bernoulli_with_abrupt_drift
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.ks_strategy import KSStrategy

# Silenzia i warning di scipy sul fallback all'approssimazione asintotica del KS
# quando applicato a dati binari (comportamento atteso, non un errore).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


# --- Parametri dell'esperimento (caricati da .env tramite config.py) ---
N_SAMPLES = BERNOULLI_N_SAMPLES
DRIFT_POINT = BERNOULLI_DRIFT_POINT
SEED = BERNOULLI_SEED
WINDOW_SIZE = KS_WINDOW_SIZE
ALPHA = KS_ALPHA

# --- Generazione delle 3 feature ---
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

# --- Setup del detector ---
detector = FeatureDriftDetector(
    strategy_cls=KSStrategy,
    n_features=3,
    k=FEATURE_K_THRESHOLD,
    feature_names=["f0_stable", "f1_drift", "f2_stable"],
    window_size=WINDOW_SIZE,
    alpha=ALPHA,
)

# --- Streaming ---
first_drift_step = None
for t in range(N_SAMPLES):
    detector.update(data[t])
    result = detector.detect()
    if result.drift_detected and first_drift_step is None:
        first_drift_step = t
        first_drift_features = result.metadata["drifted_features"]

final = detector.detect()

# --- Output su file ---
output_dir = Path("results/KS")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "KS_feature_bernoulli.txt"

with open(output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: KS Feature Drift su Bernoulli sintetica")
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
    log(f"  Detector: FeatureDriftDetector(KSStrategy)")
    log(f"  Parametri: window_size={WINDOW_SIZE}, alpha={ALPHA}, k=1")
    log()
    log("-" * 70)
    log("Risultati")
    log("-" * 70)
    if first_drift_step is None:
        log("Nessun drift rilevato durante il flusso.")
    else:
        delay = first_drift_step - DRIFT_POINT
        log(f"Primo drift rilevato al passo t = {first_drift_step}")
        log(f"  Feature drittate al primo rilevamento: {first_drift_features}")
        log(f"  Ritardo rispetto al cambio reale (t={DRIFT_POINT}): {delay} passi")
    log()
    log("Stato finale al termine del flusso:")
    log(f"  drift_detected: {final.drift_detected}")
    log(f"  drifted_features: {final.metadata['drifted_features']}")
    log(f"  n_drifted: {final.metadata['n_drifted']}")

print()
print(f"Output salvato in: {output_path}")
