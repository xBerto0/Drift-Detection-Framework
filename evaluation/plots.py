"""Generazione delle figure della validazione sperimentale.

Legge i risultati prodotti da `evaluation/runner_sintetici.py` e produce i
grafici in formato PDF vettoriale, pronti per essere inclusi in LaTeX con
\\includegraphics, piu' una copia PNG per la consultazione rapida.

Le figure si rigenerano da sole a ogni rilancio degli esperimenti: i numeri
nel testo e quelli nei grafici non possono disallinearsi.

Lancio dalla radice del progetto:
    python -m evaluation.plots
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend senza finestre, adatto all'esecuzione da script
import matplotlib.pyplot as plt


CARTELLA_RISULTATI = Path("results/sintetici")
CARTELLA_FIGURE = Path("thesis/figures")

# Palette scelta per restare leggibile anche in stampa in scala di grigi:
# colori distinti accoppiati a marcatori e tratteggi diversi.
STILE = {
    "KS":                ("#1f77b4", "o", "-"),
    "ADWIN":             ("#d62728", "s", "--"),
    "DDM":               ("#2ca02c", "^", "-."),
    "PageHinkley":       ("#9467bd", "D", ":"),
    "Ensemble-MAJORITY": ("#ff7f0e", "*", "-"),
}

ORDINE_STRATEGIE = ["KS", "ADWIN", "DDM", "PageHinkley", "Ensemble-MAJORITY"]
ORDINE_SCENARI = ["stazionario", "improvviso", "graduale", "incrementale", "ricorrente"]


def carica():
    percorso = CARTELLA_RISULTATI / "metriche_aggregate.json"
    with open(percorso, encoding="utf-8") as f:
        return json.load(f)


def salva(fig, nome):
    CARTELLA_FIGURE.mkdir(parents=True, exist_ok=True)
    for estensione in ("pdf", "png"):
        percorso = CARTELLA_FIGURE / f"{nome}.{estensione}"
        fig.savefig(percorso, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {CARTELLA_FIGURE / nome}.pdf / .png")


def figura_tradeoff(dati):
    """Latenza di rilevamento contro tasso di falsi allarmi.

    E' il grafico di riferimento nella letteratura sui drift detector: mostra
    che non esiste un detector "migliore" in assoluto, ma un compromesso fra
    prontezza e affidabilita'. In basso a sinistra sta il comportamento
    desiderabile (rileva presto, sbaglia poco).
    """
    matrice = dati["matrice"]
    scenari_con_drift = ["improvviso", "graduale", "incrementale", "ricorrente"]

    fig, assi = plt.subplots(1, 4, figsize=(15, 3.8), sharey=False)

    for ax, scenario in zip(assi, scenari_con_drift):
        for strategia in ORDINE_STRATEGIE:
            voce = matrice.get(f"{scenario}|{strategia}")
            if not voce or voce["latenza_media"] is None:
                continue
            colore, marcatore, _ = STILE[strategia]
            ax.errorbar(
                voce["falsi_allarmi_per_1000_media"],
                voce["latenza_media"],
                yerr=voce["latenza_std"],
                xerr=voce["falsi_allarmi_per_1000_std"],
                fmt=marcatore, color=colore, markersize=9,
                capsize=3, elinewidth=1, label=strategia,
            )
        ax.set_title(f"Drift {scenario}", fontsize=11)
        ax.set_xlabel("Falsi allarmi / 1000 campioni", fontsize=9)
        ax.grid(alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=8)
        # Un tasso di falsi allarmi non puo' essere negativo: le barre di
        # errore possono sconfinare sotto zero, ma l'asse no.
        ax.set_xlim(left=0)

    assi[0].set_ylabel("Latenza di rilevamento (passi)", fontsize=9)
    maniglie, etichette = assi[0].get_legend_handles_labels()
    fig.legend(maniglie, etichette, loc="lower center", ncol=5,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Compromesso fra prontezza e affidabilita' "
                 "(media su 20 ripetizioni, barre = deviazione standard)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    salva(fig, "tradeoff_latenza_falsi_allarmi")


def figura_latenza(dati):
    """Latenza media per scenario, con deviazione standard."""
    matrice = dati["matrice"]
    scenari = ["improvviso", "graduale", "incrementale", "ricorrente"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    larghezza = 0.16
    posizioni_base = range(len(scenari))

    for i, strategia in enumerate(ORDINE_STRATEGIE):
        medie, errori = [], []
        for scenario in scenari:
            voce = matrice.get(f"{scenario}|{strategia}", {})
            medie.append(voce.get("latenza_media") or 0)
            errori.append(voce.get("latenza_std") or 0)
        posizioni = [p + i * larghezza for p in posizioni_base]
        colore = STILE[strategia][0]
        ax.bar(posizioni, medie, larghezza, yerr=errori, capsize=3,
               label=strategia, color=colore, edgecolor="white", linewidth=0.5)

    ax.set_xticks([p + 2 * larghezza for p in posizioni_base])
    ax.set_xticklabels(scenari)
    ax.set_ylabel("Latenza di rilevamento (passi)")
    ax.set_title("Latenza media per tipo di drift (20 ripetizioni)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    salva(fig, "latenza_per_scenario")


def figura_mancate(dati):
    """Tasso di mancate rilevazioni per scenario."""
    matrice = dati["matrice"]
    scenari = ["improvviso", "graduale", "incrementale", "ricorrente"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    larghezza = 0.16
    posizioni_base = range(len(scenari))

    for i, strategia in enumerate(ORDINE_STRATEGIE):
        valori = []
        for scenario in scenari:
            voce = matrice.get(f"{scenario}|{strategia}", {})
            valori.append((voce.get("tasso_mancate_media") or 0) * 100)
        posizioni = [p + i * larghezza for p in posizioni_base]
        ax.bar(posizioni, valori, larghezza, label=strategia,
               color=STILE[strategia][0], edgecolor="white", linewidth=0.5)

    ax.set_xticks([p + 2 * larghezza for p in posizioni_base])
    ax.set_xticklabels(scenari)
    ax.set_ylabel("Drift non rilevati (%)")
    ax.set_title("Tasso di mancate rilevazioni per tipo di drift (20 ripetizioni)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    salva(fig, "mancate_rilevazioni")


def figura_asimmetria(dati):
    """Rilevamenti su drift in aumento contro drift in diminuzione.

    Mette in evidenza il limite di applicabilita' dei rilevatori unilaterali
    (DDM e Page-Hinkley con mode='up') e, insieme, il motivo per cui un
    ensemble ha valore: compensa i punti ciechi dei singoli membri.
    """
    asim = dati["asimmetria_direzionale"]
    direzioni = ["aumento (0.2 -> 0.6)", "diminuzione (0.6 -> 0.2)"]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    larghezza = 0.35

    for i, direzione in enumerate(direzioni):
        valori = []
        for strategia in ORDINE_STRATEGIE:
            voce = asim.get(f"{direzione}|{strategia}", {})
            totali = voce.get("n_drift_reali_totali") or 1
            valori.append(100 * (voce.get("n_rilevati_totali") or 0) / totali)
        posizioni = [j + i * larghezza for j in range(len(ORDINE_STRATEGIE))]
        colore = "#4c72b0" if i == 0 else "#c44e52"
        ax.bar(posizioni, valori, larghezza, label=direzione,
               color=colore, edgecolor="white", linewidth=0.5)

    ax.set_xticks([j + larghezza / 2 for j in range(len(ORDINE_STRATEGIE))])
    ax.set_xticklabels(ORDINE_STRATEGIE, fontsize=8)
    ax.set_ylabel("Drift rilevati (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Asimmetria direzionale: i rilevatori unilaterali non vedono i cali")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    salva(fig, "asimmetria_direzionale")


def figura_falsi_allarmi_stazionario(dati):
    """Falsi allarmi sullo stream stazionario, dove ogni segnalazione e' errata."""
    matrice = dati["matrice"]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    valori, errori, colori = [], [], []
    for strategia in ORDINE_STRATEGIE:
        voce = matrice.get(f"stazionario|{strategia}", {})
        valori.append(voce.get("falsi_allarmi_per_1000_media") or 0)
        errori.append(voce.get("falsi_allarmi_per_1000_std") or 0)
        colori.append(STILE[strategia][0])

    ax.bar(range(len(ORDINE_STRATEGIE)), valori, 0.6, yerr=errori, capsize=4,
           color=colori, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(ORDINE_STRATEGIE)))
    ax.set_xticklabels(ORDINE_STRATEGIE, fontsize=8)
    ax.set_ylabel("Falsi allarmi / 1000 campioni")
    ax.set_title("Falsi allarmi su stream stazionario (nessun drift presente)")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    salva(fig, "falsi_allarmi_stazionario")


def main():
    dati = carica()
    print("Generazione delle figure:")
    figura_tradeoff(dati)
    figura_latenza(dati)
    figura_mancate(dati)
    figura_asimmetria(dati)
    figura_falsi_allarmi_stazionario(dati)
    print(f"\nFigure pronte per LaTeX in {CARTELLA_FIGURE}/")


if __name__ == "__main__":
    main()
