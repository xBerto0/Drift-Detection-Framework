"""Carica i parametri di configurazione dal file .env.

Espone le variabili con i tipi corretti (int / float / str) agli script
sperimentali. Se .env non e' presente o una variabile e' mancante, usa i
valori di default definiti qui.
"""

import os

from dotenv import load_dotenv

# Carica le variabili dal file .env nel processo corrente.
load_dotenv()


# --- Dataset ---
DATASET_ELEC2_PATH = os.getenv(
    "DATASET_ELEC2_PATH", "data/electricity-normalized.csv"
)


# --- Esperimento Bernoulli sintetico ---
BERNOULLI_N_SAMPLES = int(os.getenv("BERNOULLI_N_SAMPLES", 2000))
BERNOULLI_DRIFT_POINT = int(os.getenv("BERNOULLI_DRIFT_POINT", 1000))
BERNOULLI_SEED = int(os.getenv("BERNOULLI_SEED", 42))


# --- Esperimento ELEC2 ---
ELEC2_TRAIN_SIZE = int(os.getenv("ELEC2_TRAIN_SIZE", 500))


# --- Strategia KS ---
KS_WINDOW_SIZE = int(os.getenv("KS_WINDOW_SIZE", 200))
KS_ALPHA = float(os.getenv("KS_ALPHA", 0.05))


# --- Strategia ADWIN ---
ADWIN_DELTA = float(os.getenv("ADWIN_DELTA", 0.002))


# --- FeatureDriftDetector ---
FEATURE_K_THRESHOLD = int(os.getenv("FEATURE_K_THRESHOLD", 1))
