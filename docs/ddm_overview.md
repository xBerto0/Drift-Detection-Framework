# DDM — Drift Detection Method

**Paper di riferimento**: Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004).
*Learning with Drift Detection*. In: Advances in Artificial Intelligence — SBIA 2004,
Lecture Notes in Computer Science, vol. 3171, pp. 286–295. Springer.
DOI: `10.1007/978-3-540-28645-5_29`

**Implementazione usata**: `river.drift.binary.DDM` (river 0.22.0)
**Wrapper nel framework**: `detectors/ddm_strategy.py`

---

## 1. Il problema che risolve

KS e ADWIN guardano i **dati**. DDM guarda il **modello**.

È una differenza importante, e vale la pena averla chiara perché è il primo
argomento che ti verrà chiesto in discussione:

- **KS** confronta due finestre di una feature e dice se la distribuzione è cambiata.
  Non sa nemmeno che esiste un modello.
- **ADWIN** monitora la media di uno stream qualsiasi. Anche lui è agnostico
  rispetto al modello.
- **DDM** monitora il **tasso di errore del classificatore**. Non gli interessa
  come sono fatti i dati: gli interessa se il modello sta sbagliando di più.

Questa è la distinzione fra rilevamento **non supervisionato** (KS, ADWIN sulle
feature: non servono le etichette) e **supervisionato** (DDM: servono le `y_true`,
perché senza non sai se il modello ha sbagliato).

Il vantaggio di DDM è che rileva **esattamente ciò che ti interessa davvero**: il
degrado del modello. Una feature può cambiare distribuzione senza che il modello
peggiori — sarebbe un falso allarme dal punto di vista operativo. DDM per
costruzione non può avere quel tipo di falso allarme.

Lo svantaggio è il prezzo: **servono le etichette vere**, e in produzione spesso
arrivano tardi o non arrivano affatto (il *label delay* di cui parli nel Cap. 5).

---

## 2. L'intuizione

L'idea di partenza è una sola frase, ed è quella da ricordare:

> Se il concetto è stazionario, il tasso di errore di un classificatore che
> apprende deve **calare o restare stabile** man mano che vede più esempi.
> Se ricomincia a **salire in modo significativo**, il concetto è cambiato.

È un'applicazione del **controllo statistico di processo** (SPC), le stesse carte
di controllo che si usano in produzione industriale per capire se una linea di
montaggio sta andando fuori tolleranza. Gama lo dice esplicitamente nel paper: il
learner è visto come un processo, e l'errore è la metrica di qualità del processo.

---

## 3. La formalizzazione

Trattiamo l'esito di ogni predizione come una variabile di **Bernoulli**:
1 se il modello ha sbagliato, 0 se ha indovinato.

Dopo `i` esempi:

- **p_i** = tasso di errore osservato = (numero di errori) / i
- **s_i** = deviazione standard = `sqrt( p_i * (1 - p_i) / i )`

La formula di `s_i` è la deviazione standard della media campionaria di una
Bernoulli. Nota il comportamento: **al crescere di `i`, `s_i` si restringe**.
Più esempi hai, più sei sicuro della tua stima del tasso di errore.

DDM tiene in memoria due valori:

- **p_min**, **s_min** — la coppia (p, s) osservata nell'istante in cui la somma
  `p_i + s_i` ha toccato il suo **minimo storico**.

Il minimo storico rappresenta *"il momento in cui il modello era al suo meglio"*.
Ogni volta che `p_i + s_i` scende sotto il minimo registrato, il minimo si aggiorna.

Poi confronta il valore corrente con due soglie:

```
  p_i + s_i  >=  p_min + 2 * s_min      ->  livello di WARNING
  p_i + s_i  >=  p_min + 3 * s_min      ->  livello di DRIFT
```

I coefficienti 2 e 3 corrispondono, sotto approssimazione normale, a circa il
95% e il 99% di confidenza. Sono gli stessi coefficienti delle carte di controllo
classiche (limiti a 2σ e 3σ).

**Perché si usa `p + s` e non solo `p`?** Perché sommare la deviazione standard
rende il test **automaticamente più prudente quando hai pochi dati**: all'inizio
`s` è grande, quindi la soglia è alta e si evita di gridare al drift sul rumore.
Man mano che accumuli esempi `s` si restringe e il test diventa più sensibile.
È una forma elegante di autoregolazione.

---

## 4. Pseudocodice

```
per ogni nuova predizione:
    aggiorna p_i (tasso di errore corrente) e calcola s_i

    se i < warm_start:
        non fare nulla        # troppo pochi dati per una baseline affidabile

    se p_i + s_i < p_min + s_min:
        p_min <- p_i
        s_min <- s_i          # nuovo "momento migliore" del modello

    se p_i + s_i >= p_min + 3*s_min:
        segnala DRIFT
        reimposta tutte le statistiche   # si riparte da zero sul nuovo concetto

    altrimenti se p_i + s_i >= p_min + 2*s_min:
        segnala WARNING

    altrimenti:
        stato normale
```

---

## 5. Le due soglie: perché sono il punto centrale per la tua tesi

Questo è **il motivo principale per cui abbiamo scelto DDM** invece di un'altra
variante, ed è l'argomento da portare al tutor aziendale.

Nessuna delle strategie che avevi (KS, ADWIN) ha una nozione di *warning*.
Segnalano drift oppure no. DDM invece ha uno **stato intermedio**, e quello stato
risolve un problema pratico che altrimenti non ha soluzione pulita:

> Quando decidi che c'è drift e vuoi riaddestrare il modello, **su quali dati lo
> riaddestri?** I dati vecchi appartengono al vecchio concetto e sono inutili o
> dannosi. I dati nuovi sono pochissimi, perché il drift l'hai appena rilevato.

Il meccanismo di DDM risolve così:

1. **Entra in WARNING** → si comincia ad accumulare i campioni in un buffer separato
2. **Il drift viene CONFERMATO** → si riaddestra il modello **sul buffer**, che
   contiene già campioni del nuovo concetto raccolti durante il warning
3. **L'allarme rientra senza conferma** → il buffer si scarta e si continua

Questo è esattamente l'**Objective 4** della tua proposta ("Identify retraining
triggers based on detected drift"). DDM non è un algoritmo in più: è il pezzo che
rende implementabile il capitolo sul retraining.

---

## 6. Limiti (da scrivere nella tesi, non da nascondere)

1. **Richiede le etichette vere.** Senza `y_true` non calcoli l'errore. In
   produzione con label delay, DDM segnala il drift con il ritardo con cui
   arrivano le etichette, non con la latenza dell'algoritmo.

2. **Solo classificazione — ma è un limite dell'implementazione, non del
   metodo.** `river` deriva `s_i` dalla binomiale e richiede quindi un errore
   binario. Gama et al. (2004), nelle conclusioni, prospettano invece l'uso
   dell'algoritmo con qualunque funzione di perdita e riportano risultati
   preliminari in regressione con l'errore quadratico medio. Vedere
   `docs/ddm_riferimenti_scientifici.md`, §5.4.

3. **Rileva solo il drift che peggiora l'errore.** Se il concetto cambia ma il
   modello continua a indovinare (per fortuna o perché il cambiamento è
   irrilevante per la decisione), DDM non vede nulla. Dal punto di vista
   operativo è un pregio; dal punto di vista della caratterizzazione del
   fenomeno è un limite.

4. **Lento sui drift graduali.** Su un drift molto lento, `p_min + s_min`
   viene aggiornato progressivamente e la baseline "insegue" il degrado. È il
   problema che EDDM (Baena-García et al. 2006) cerca di risolvere monitorando
   la *distanza fra due errori consecutivi* anziché il tasso.
   ⚠️ Attenzione ad attribuire questo limite al paper originale: **non lo
   dichiara**, e include anzi un dataset a drift graduale (CIRCLES).
   L'affermazione proviene dagli autori di EDDM. Vedere
   `docs/ddm_riferimenti_scientifici.md`, §4.

5. **Assume che il learner migliori.** L'ipotesi di partenza è un modello che
   apprende online. Nel nostro esperimento su ELEC2 il classificatore è
   **statico** (addestrato una volta e mai aggiornato): l'ipotesi è violata.
   Va detto esplicitamente in tesi — è una scelta consapevole che semplifica
   l'esperimento, non una svista.

---

## 7. Come si integra nel framework

DDM è una **strategia** (livello 1 dell'architettura), come KS e ADWIN. Non ha
richiesto alcuna modifica a `BaseDriftDetector`, `ConceptDriftDetector` o
`EnsembleStrategy`: è la terza conferma empirica della bontà del pattern Strategy.

```python
from detectors.concept_drift_detector import ConceptDriftDetector
from detectors.ddm_strategy import DDMStrategy

detector = ConceptDriftDetector(DDMStrategy, warm_start=200)

detector.update(y_pred, y_true)
risultato = detector.detect()

risultato.drift_detected              # True/False
risultato.metadata["warning_detected"]  # lo stato intermedio
```

Una sola differenza rispetto a KS e ADWIN: `DDMStrategy.update()` **solleva un
errore** se riceve un valore non binario. È una scelta deliberata: DDM applicato
a una feature continua produrrebbe numeri privi di significato, e un errore
esplicito è meglio di un risultato silenziosamente sbagliato.

---

## 8. Il risultato sperimentale già ottenuto sul parametro `warm_start`

Nel primo test su stream sintetico (errore 10% fino a t=1000, poi 50%),
ripetuto su 5 seed diversi:

| `warm_start` | Seed con falsi allarmi | Latenza di rilevamento |
|---|---|---|
| **30** (default di river) | 2 su 5 (a t=173 e t=114) | 32–79 passi |
| **100** | 1 su 5 (due falsi allarmi) | 11–59 passi |
| **200** | **0 su 5** | 27–59 passi |

**Interpretazione**: con `warm_start=30` la baseline `p_min + s_min` viene fissata
su troppi pochi campioni. Se quei 30 campioni contengono per caso pochi errori, il
minimo storico risulta artificialmente basso e il normale ritorno alla media viene
scambiato per un drift. Con 200 campioni la baseline è stabile e i falsi allarmi
spariscono, **senza peggiorare la latenza**.

Per questo nel `.env` è impostato `DDM_WARM_START=200` anziché il default della
libreria.

Due cose da notare, perché valgono più del risultato in sé:

- Con **un solo seed** (il 42) avresti concluso *"DDM ha un falso positivo a
  t=173"*. Con cinque seed vedi che dipende dal seed ed è eliminabile. È la
  dimostrazione, sui tuoi stessi dati, del perché la validazione deve essere
  multi-seed.
- È una **scelta di parametro giustificata sperimentalmente**, non copiata da
  un default. In tesi vale molto più di una tabella di risultati.

---

## 9. Domande probabili in discussione, e come rispondere

**"Perché DDM e non semplicemente ADWIN sull'errore?"**
Sono complementari. ADWIN è generico e non fa assunzioni: monitora la media di
qualsiasi stream limitato, con garanzie teoriche formali (Teorema 3.1). DDM è
specializzato: assume esplicitamente il modello Bernoulli dell'errore e sfrutta
quell'assunzione per fornire il livello di *warning*, che ADWIN non ha. Se ti
serve solo sapere *se* c'è drift, ADWIN basta. Se ti serve *gestire* il drift
riaddestrando, il warning di DDM è ciò che rende la cosa praticabile.

**"Perché i coefficienti 2 e 3?"**
Sono i limiti di controllo classici dell'SPC, corrispondenti a circa 95% e 99%
di confidenza sotto approssimazione normale della binomiale. Sono i valori del
paper originale, e nel framework restano configurabili.

**"DDM funziona sul concept drift o sul data drift?"**
Solo concept drift, e solo nella sua accezione *supervisionata*: rileva un
cambiamento in `P(y|X)` osservandone l'effetto sull'errore. Non vede un
cambiamento di `P(X)` che non degradi le prestazioni.

**"Nel vostro esperimento il classificatore è statico. DDM non assume un
learner che apprende?"**
Sì, ed è una violazione consapevole dell'ipotesi. Con un modello statico
l'errore non cala mai, quindi `p_min + s_min` si stabilizza presto e DDM di
fatto rileva ogni peggioramento rispetto alla prestazione iniziale. Il
comportamento resta sensato per il nostro scopo, ma la scelta va dichiarata.

---

## 10. Riferimenti

- **Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004).** *Learning with
  Drift Detection*. SBIA 2004, LNCS 3171, pp. 286–295. Springer.
  DOI `10.1007/978-3-540-28645-5_29` — **paper originale**
- **Baena-García, M. et al. (2006).** *Early Drift Detection Method*. ECML PKDD
  Workshop on Knowledge Discovery from Data Streams — la variante EDDM
- **Lu, J. et al. (2019).** *Learning under Concept Drift: A Review*. IEEE TKDE
  — descrive DDM nel contesto generale (già presente in `Papers/`)
- **Gama, J. et al. (2014).** *A Survey on Concept Drift Adaptation*. ACM
  Computing Surveys — inquadramento SPC (già presente in `Papers/`)
- **Montiel, J. et al. (2021).** *River: machine learning for streaming data in
  Python*. JMLR — la libreria usata
