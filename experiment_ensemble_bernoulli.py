"""Test dell'EnsembleStrategy (KS + ADWIN) su feature Bernoulli sintetiche.

Lancia due detector in parallelo con lo stesso ensemble ma regole di
aggregazione diverse (OR e AND) per mostrare sperimentalmente come cambia
il comportamento al variare della regola.

Configurazione parametri caricata da .env tramite config.py.
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    ADWIN_DELTA,
    BERNOULLI_DRIFT_POINT,
    BERNOULLI_N_SAMPLES,
    BERNOULLI_SEED,
    FEATURE_K_THRESHOLD,
    KS_ALPHA,
    KS_WINDOW_SIZE,
)
from data.synthetic_generator import bernoulli_with_abrupt_drift
from detectors.adwin_strategy import ADWINStrategy
from detectors.ensemble_strategy import EnsembleStrategy
from detectors.feature_drift_detector import FeatureDriftDetector
from detectors.ks_strategy import KSStrategy

# Silenzia i warning di scipy sul fallback all'approssimazione asintotica del KS
# quando applicato a dati binari (comportamento atteso, non un errore).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


# --- Parametri dell'esperimento ---
N_SAMPLES = BERNOULLI_N_SAMPLES
DRIFT_POINT = BERNOULLI_DRIFT_POINT
SEED = BERNOULLI_SEED

# --- Composizione dell'ensemble (uguale per OR e AND) ---
ENSEMBLE_SPECS = [
    (KSStrategy, {"window_size": KS_WINDOW_SIZE, "alpha": KS_ALPHA}),
    (ADWINStrategy, {"delta": ADWIN_DELTA}),
]

# --- Generazione delle 3 feature Bernoulli ---
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
feature_names = ["f0_stable", "f1_drift", "f2_stable"]


# --- Setup di due detector con aggregazione diversa ---
detector_or = FeatureDriftDetector(
    strategy_cls=EnsembleStrategy,
    n_features=3,
    k=FEATURE_K_THRESHOLD,
    feature_names=feature_names,
    strategy_specs=ENSEMBLE_SPECS,
    aggregation="OR",
)

detector_and = FeatureDriftDetector(
    strategy_cls=EnsembleStrategy,
    n_features=3,
    k=FEATURE_K_THRESHOLD,
    feature_names=feature_names,
    strategy_specs=ENSEMBLE_SPECS,
    aggregation="AND",
)


# --- Streaming ---
# Per ciascun detector traccio i punti di drift e per ciascuna feature
# quale sotto-strategia ha contribuito.
drift_steps_or = []
drift_steps_and = []
detailed_verdicts_or = []
detailed_verdicts_and = []


def collect_verdicts(feature_detector, t):
    """Interroga le sotto-strategie interne all'ensemble di ogni feature.

    Restituisce una lista di dict {feature: name, KS: bool, ADWIN: bool}
    solo per le feature che sono attualmente in drift secondo l'aggregato.
    """
    result = feature_detector.detect()
    if not result.drift_detected:
        return None

    per_feature = []
    drifted_names = set(result.metadata["drifted_features"])
    for i, ensemble in enumerate(feature_detector.strategies):
        feature_name = feature_names[i]
        if feature_name not in drifted_names:
            continue
        # ensemble e' un'istanza di EnsembleStrategy: le sue sub-strategie
        # sono in ensemble.strategies
        sub_verdicts = {}
        for sub in ensemble.strategies:
            sub_result = sub.detect()
            sub_verdicts[sub.detector_name] = sub_result.drift_detected
        per_feature.append({"feature": feature_name, "verdicts": sub_verdicts})
    return per_feature


for t in range(N_SAMPLES):
    x_t = data[t]
    detector_or.update(x_t)
    detector_and.update(x_t)

    # OR
    per_feat = collect_verdicts(detector_or, t)
    if per_feat is not None:
        drift_steps_or.append(t)
        detailed_verdicts_or.append((t, per_feat))

    # AND
    per_feat = collect_verdicts(detector_and, t)
    if per_feat is not None:
        drift_steps_and.append(t)
        detailed_verdicts_and.append((t, per_feat))


# --- Output su file ---
output_dir = Path("results/ENSEMBLE")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "ENSEMBLE_feature_bernoulli.txt"

with open(output_path, "w", encoding="utf-8") as f:
    def log(msg=""):
        print(msg)
        f.write(msg + "\n")

    log("=" * 70)
    log("Esperimento: Ensemble (KS + ADWIN) Feature Drift su Bernoulli sintetica")
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
    log(f"  Strategie interne all'ensemble:")
    log(f"    - KSStrategy(window_size={KS_WINDOW_SIZE}, alpha={KS_ALPHA})")
    log(f"    - ADWINStrategy(delta={ADWIN_DELTA})")
    log()
    log(f"  Test in parallelo con aggregazione OR e AND")
    log()

    # --- Sezione OR ---
    log("-" * 70)
    log("Risultati con aggregazione OR (drift se almeno una segnala)")
    log("-" * 70)
    log(f"Passi totali di drift: {len(drift_steps_or)}")
    if drift_steps_or:
        log(f"Primo drift: t={drift_steps_or[0]}")
        log(f"Ultimo drift: t={drift_steps_or[-1]}")
        log()
        log("Prime 10 segnalazioni con dettaglio delle sotto-strategie:")
        for t, per_feat in detailed_verdicts_or[:10]:
            for entry in per_feat:
                feat = entry["feature"]
                v = entry["verdicts"]
                log(f"  t={t:4d}  {feat}  KS={v.get('KS')!s:5}  ADWIN={v.get('ADWIN')!s:5}")
    log()

    # --- Sezione AND ---
    log("-" * 70)
    log("Risultati con aggregazione AND (drift solo se tutte segnalano)")
    log("-" * 70)
    log(f"Passi totali di drift: {len(drift_steps_and)}")
    if drift_steps_and:
        log(f"Primo drift: t={drift_steps_and[0]}")
        log(f"Ultimo drift: t={drift_steps_and[-1]}")
        log()
        log("Dettaglio di tutte le segnalazioni (attesi pochi punti):")
        for t, per_feat in detailed_verdicts_and:
            for entry in per_feat:
                feat = entry["feature"]
                v = entry["verdicts"]
                log(f"  t={t:4d}  {feat}  KS={v.get('KS')!s:5}  ADWIN={v.get('ADWIN')!s:5}")
    log()

    # --- Confronto e riepilogo ---
    log("-" * 70)
    log("Confronto OR vs AND")
    log("-" * 70)
    log(f"Numero segnalazioni OR:  {len(drift_steps_or)}")
    log(f"Numero segnalazioni AND: {len(drift_steps_and)}")
    if drift_steps_or and drift_steps_and:
        log(f"Primo drift OR:  t={drift_steps_or[0]}")
        log(f"Primo drift AND: t={drift_steps_and[0]}")
        log(f"Differenza fra i due primi drift: {drift_steps_and[0] - drift_steps_or[0]} passi")
    log()
    log("Interpretazione attesa:")
    log("  - OR e' sensibile: appena UNA strategia segnala, scatta.")
    log("    Su Bernoulli con KS a continuo signaling, OR imita KS.")
    log("  - AND e' conservativo: entrambe devono concordare.")
    log("    ADWIN segnala solo al momento del taglio della finestra,")
    log("    quindi AND scatta solo quando anche KS sta gia' segnalando.")


print()
print(f"Output salvato in: {output_path}")
