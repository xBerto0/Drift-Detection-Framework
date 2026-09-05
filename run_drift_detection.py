"""Punto di ingresso del servizio di monitoring.

Legge la configurazione, esegue i rilevamenti richiesti e scrive l'esito in
formato JSON.

    python run_drift_detection.py
    python run_drift_detection.py --config altra_config.json --output esito.json

Il percorso predefinito della configurazione si legge dal file .env, voce
DRIFT_CONFIG_PATH.
"""

import argparse
import json
import sys

from config import DRIFT_CONFIG_PATH
from monitoring.drift_monitoring_service import (
    DriftMonitoringService, ErroreConfigurazione,
)


def main():
    parser = argparse.ArgumentParser(
        description="Rileva il drift secondo una configurazione dichiarativa.",
    )
    parser.add_argument(
        "--config", default=DRIFT_CONFIG_PATH,
        help="File di configurazione JSON (default: da .env)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Dove scrivere l'esito. Se omesso, stampa a video.",
    )
    argomenti = parser.parse_args()

    try:
        servizio = DriftMonitoringService.da_file(argomenti.config)
        esito = servizio.esegui()
    except ErroreConfigurazione as e:
        print(f"Configurazione non valida: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"File non trovato: {e}", file=sys.stderr)
        return 1

    testo = json.dumps(esito, indent=2, ensure_ascii=False)
    if argomenti.output:
        with open(argomenti.output, "w", encoding="utf-8") as f:
            f.write(testo + "\n")
        print(f"Esito scritto in {argomenti.output}")
    else:
        print(testo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
