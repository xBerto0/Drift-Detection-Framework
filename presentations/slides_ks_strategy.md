---
marp: true
theme: default
paginate: true
size: 16:9
---

# Drift Detection con KS

Implementazione della strategia di Kolmogorov-Smirnov nel framework di drift detection.

Alberto Conti

---

# Cosa fa il KS

Il KS test è un test statistico che confronta due gruppi di numeri e dice se vengono dalla stessa distribuzione.

Nel mio caso confronta:
- una **reference window**: i dati "normali", quelli che si vedono all'inizio
- una **current window**: gli ultimi dati arrivati

Se sono troppo diverse → drift.

---

# Che tipo di drift gestisce

Lo uso per il **feature drift**: monitora la distribuzione delle feature in input al modello, indipendentemente dal modello stesso (è model-agnostic).

Lo stesso algoritmo si potrà applicare anche alle predizioni del modello quando passeremo al *prediction drift*.

---

# Come funziona, in 4 passi

1. Prendo le due finestre (reference e current)
2. Costruisco la **ECDF** di ciascuna (la curva cumulativa della distribuzione)
3. Misuro la **massima distanza verticale** tra le due → statistica D
4. Da D ricavo il **p-value**

Se il p-value scende sotto la soglia α → dichiaro drift.

---

# Cos'è il p-value

Risponde a questa domanda:

> "Se le due finestre venissero davvero dalla stessa distribuzione, quanto sarebbe stato improbabile vedere una differenza così grande per puro caso?"

- p-value alto → tutto normale
- p-value basso → differenza troppo grande per essere caso → drift

Soglia tipica: **α = 0.05**.

---

# Implementazione: `KSStrategy`

Una classe semplice che estende `BaseDriftDetector`.

Dentro:
- due `deque` a lunghezza fissa: `reference` e `current`
- `update(value)`: riempie le finestre (prima reference, poi current scorrevole)
- `detect()`: esegue `scipy.stats.ks_2samp(reference, current)` e restituisce un `DriftResult`

Una `KSStrategy` lavora su **un singolo stream** di numeri.

---

# Implementazione: `FeatureDriftDetector`

Per monitorare più feature insieme serve un orchestratore.

Il `FeatureDriftDetector`:
- riceve un **vettore** (es. 5 feature)
- contiene **N istanze** di `KSStrategy`, una per feature
- smista i valori, aggrega i verdetti
- dichiara drift se almeno **k feature** sono in drift (k configurabile)

Domani posso sostituire `KSStrategy` con un'altra strategia senza toccare il resto.

---

# Test fatto

Esperimento sintetico fedele al paper di Bifet & Gavaldà (2006):
- 3 feature Bernoulli, 2000 campioni
- 2 feature stabili + 1 feature con cambio brusco al passo 1000

**Risultato:** il detector ha identificato correttamente la sola feature drittata, **senza falsi positivi** sulle altre. Ritardo di rilevamento ~100 passi, dovuto al tempo necessario alla current window per riempirsi di valori nuovi.

---

# Prossimi passi

- KS su dati binari ha potere statistico ridotto (KS è pensato per distribuzioni continue) → prossimo: **ADWIN**, nativo per i binari e con finestre adattive
- Test su un **dataset reale** (ELEC2, lo stesso del paper)
- Estendere a *prediction drift* riutilizzando le stesse strategie

---

# Grazie

Domande?
