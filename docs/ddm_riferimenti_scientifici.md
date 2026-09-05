# DDM — Drift Detection Method: fondamenti dalla letteratura

Documento di riferimento scientifico su DDM. Ogni affermazione tecnica riporta
la fonte da cui proviene. È il materiale destinato al Capitolo 4 della tesi.

Per la parte pratica — integrazione nel framework, parametri esposti, risultati
sperimentali ottenuti — vedere `docs/ddm_overview.md`, che è complementare a
questo e non lo sostituisce.

---

## Nota preliminare sulle fonti

**Il paper primario non è liberamente accessibile.**

> Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004). *Learning with Drift
> Detection*. In: Bazzan, A.L.C., Labidi, S. (eds) Advances in Artificial
> Intelligence — SBIA 2004. Lecture Notes in Computer Science, vol. 3171,
> pp. 286–295. Springer, Berlin/Heidelberg.
> DOI: [10.1007/978-3-540-28645-5_29](https://doi.org/10.1007/978-3-540-28645-5_29)

Lo stato di accesso è stato verificato tramite Unpaywall: `is_oa: false`,
`oa_status: "closed"`, nessuna versione ad accesso aperto disponibile. **Va
recuperato tramite l'accesso istituzionale Unipa a SpringerLink** (oppure
tramite l'abbonamento aziendale di Engineering) prima della consegna della
tesi: un capitolo che descrive DDM deve poter citare il paper originale.

Nel frattempo, quanto segue è ricostruito da **fonti secondarie sottoposte a
revisione paritaria che descrivono formalmente l'algoritmo**, tutte verificabili
e due delle quali già presenti in `Papers/`. Dove una formula o una definizione
proviene da una di queste fonti, la fonte è indicata esplicitamente.

### Fonti effettivamente consultate

| Sigla | Riferimento | Disponibilità |
|---|---|---|
| **[SK17]** | Sethi, T. S., Kantardzic, M. (2017). *On the reliable detection of concept drift from streaming unlabeled data*. Expert Systems with Applications, 82, 77–99. [arXiv:1704.00023](https://arxiv.org/abs/1704.00023) | in `Papers/`, e accesso aperto |
| **[LU19]** | Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2019). *Learning under Concept Drift: A Review*. IEEE Transactions on Knowledge and Data Engineering, 31(12), 2346–2363. [arXiv:2004.05785](https://arxiv.org/abs/2004.05785) | in `Papers/`, e accesso aperto |
| **[GA14]** | Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). *A Survey on Concept Drift Adaptation*. ACM Computing Surveys, 46(4), 1–37. | in `Papers/` ⚠️ vedere nota |
| **[BG07]** | Bifet, A., Gavaldà, R. (2007). *Learning from Time-Changing Data with Adaptive Windowing*. SIAM Int. Conf. on Data Mining, pp. 443–448. | in `Papers/` |

> ⚠️ **Nota su [GA14]**: la copia presente in `Papers/` è una versione italiana
> di 17 pagine, mentre l'originale ACM ne conta circa 37. Non contiene alcuna
> occorrenza di DDM. Per la bibliografia della tesi serve l'originale inglese
> completo.

---

## 1. Collocazione tassonomica

[SK17] classifica i metodi di rilevamento del drift in **espliciti**
(supervisionati) e **impliciti** (non supervisionati), e all'interno degli
espliciti distingue tre famiglie:

| Famiglia | Metodi | Riferimenti in [SK17] |
|---|---|---|
| Analisi sequenziale | CUSUM, PHT, LFR | Page 1954; Wang e Abraham 2015 |
| **Controllo statistico di processo** | **DDM**, EDDM, STEPD, EWMA | **Gama et al. 2004**; Baena-García et al. 2006; Nishida e Yamauchi 2007; Ross et al. 2012 |
| Monitoraggio della distribuzione a finestre | ADWIN, DoD, Resampling | Bifet e Gavaldà 2007; Sobhani e Beigy 2011; Harel et al. 2014 |

**DDM appartiene quindi alla famiglia del controllo statistico di processo**
(SPC), non a quella dell'analisi sequenziale a cui appartiene Page-Hinkley, né
a quella del windowing adattivo a cui appartiene ADWIN. È la distinzione che
giustifica la scelta di includere nel framework un rappresentante per ciascuna
delle tre famiglie.

Sempre secondo [SK17], le tecniche SPC monitorano l'andamento online del tasso
di errore e rilevano scostamenti applicando idee mutuate dalle **carte di
controllo** usate nel controllo qualità industriale.

---

## 2. Il presupposto teorico: il modello PAC

Sia [SK17] sia [LU19] individuano il fondamento teorico di DDM nel **modello di
apprendimento PAC** (Probably Approximately Correct).

Il ragionamento riportato da [SK17] è il seguente: nel modello PAC, se la
distribuzione che genera i dati è **stazionaria**, il tasso di errore di un
learner deve diminuire — o al più restare stabile — al crescere del numero di
esempi osservati. Ne discende la conclusione operativa:

> un aumento significativo del tasso di errore **viola il modello PAC**, e viene
> quindi assunto come conseguenza di un concept drift.
> *(parafrasato da [SK17], §2.1.2)*

Questo è il punto concettuale centrale, e vale la pena esplicitarlo perché è
ciò che distingue DDM da un generico rilevatore di cambiamento di media: DDM
**non** osserva i dati, osserva la violazione di una garanzia teorica sul
comportamento del learner.

### Conseguenza da dichiarare in tesi

Il presupposto è che il learner **apprenda**. Negli esperimenti condotti in
questo lavoro il classificatore è **statico**, addestrato una volta e mai
aggiornato: l'ipotesi PAC di errore decrescente non è quindi soddisfatta. Il
comportamento resta interpretabile — con un modello statico il tasso di errore
si stabilizza presto e DDM segnala ogni peggioramento rispetto a quella
baseline — ma la deviazione dall'assunzione originale va dichiarata
esplicitamente.

---

## 3. La formulazione

La formulazione seguente è riportata **testualmente** in [SK17], §2.1.2.

Sia `p_t` la probabilità di errore osservata al tempo `t` e `i` il numero di
esempi visti. La deviazione standard associata è:

```
    s_t = sqrt( p_t * (1 - p_t) / i )
```

È la deviazione standard della media campionaria di una variabile di Bernoulli:
l'esito di ciascuna predizione viene trattato come una prova bernoulliana
(errore / non errore).

L'algoritmo mantiene due registri. Citando [SK17]:

> quando `p_t + s_t` raggiunge il suo valore minimo, i valori corrispondenti
> vengono memorizzati in `p_min` e `s_min`.

Le due soglie sono:

```
    p_t + s_t  >=  p_min + 2 * s_min      ->   WARNING
    p_t + s_t  >=  p_min + 3 * s_min      ->   DRIFT
```

La stessa formulazione è confermata in modo indipendente dalla letteratura
successiva, che descrive l'aggiornamento dei registri come effettuato quando
`p_i + s_i < p_min + s_min`.

### Perché la somma `p + s` e non il solo `p`

Nessuna delle fonti consultate lo esplicita, ma la conseguenza è verificabile
direttamente dalla formula ed è un'osservazione difendibile in sede di
discussione: poiché `s_t` decresce come `1/sqrt(i)`, all'inizio dello stream la
soglia è ampia e il test è prudente; man mano che si accumulano esempi la
soglia si stringe e il test diventa più sensibile. La statistica di test
**autoregola la propria severità in funzione della quantità di evidenza
disponibile**.

I coefficienti 2 e 3 corrispondono ai limiti di controllo classici delle carte
SPC, cioè a circa il 95% e il 99% di confidenza sotto approssimazione normale
della binomiale — coerentemente con la collocazione di DDM nella famiglia SPC
data da [SK17].

---

## 4. I due livelli di allarme: il contributo distintivo

[LU19] identifica esplicitamente in questo il contributo storico di DDM:

> DDM è stato **il primo algoritmo a definire il livello di warning e il livello
> di drift** per il rilevamento del concept drift.
> *(parafrasato da [LU19], §3)*

[LU19] descrive anche il **protocollo operativo** che i due livelli rendono
possibile, ed è la parte più rilevante per la parte MLOps di questo lavoro:

1. quando la confidenza sul cambiamento del tasso di errore raggiunge il
   **livello di warning**, DDM comincia a costruire un nuovo learner,
   continuando nel frattempo a usare quello vecchio per le predizioni;
2. quando si raggiunge il **livello di drift**, il vecchio learner viene
   sostituito da quello nuovo.

Questo risolve un problema pratico che altrimenti non ha soluzione pulita:
**su quali dati riaddestrare?** I dati vecchi appartengono al concetto
superato; quelli nuovi, nell'istante in cui il drift viene confermato, sarebbero
pochissimi. La zona di warning è la finestra temporale durante la quale si
accumulano campioni che appartengono già al nuovo concetto.

### Collocazione nel framework a quattro stadi di [LU19]

[LU19] propone un framework generale in quattro stadi per il rilevamento del
drift, e vi colloca DDM così:

| Stadio | In DDM |
|---|---|
| 1 — Raggruppamento dei dati | Finestra temporale **landmark**: il punto iniziale è fisso, quello finale si estende a ogni nuova istanza |
| 2 — Modellazione dei dati | Il classificatore che produce le predizioni |
| 3 — Statistica di test | Il tasso di errore online |
| 4 — Test di ipotesi | Stima della distribuzione del tasso di errore e calcolo delle soglie di warning e di drift |

La natura **landmark** della finestra è un dettaglio importante e spesso
trascurato: DDM non usa una finestra scorrevole. Accumula dal punto di
partenza, e questo spiega perché dopo un drift l'algoritmo debba essere
reinizializzato.

---

## 5. Limiti documentati in letteratura

### 5.1 Il drift graduale

È il limite riconosciuto dalle fonti in modo più esplicito. [SK17] afferma che
EDDM (Baena-García et al., 2006)

> è stato sviluppato come estensione di DDM ed è stato reso adatto ai drift
> graduali lenti, **dove DDM in precedenza falliva**.
> *(parafrasato da [SK17], §2.1.2)*

EDDM cambia la statistica monitorata: invece del tasso di errore osserva la
**distanza, in numero di campioni, fra due errori di classificazione
consecutivi**. Sotto il modello PAC, in ambiente stazionario ci si attende che
tale distanza cresca.

Il limite è coerente con quanto misurato negli esperimenti di questo lavoro: sul
drift graduale e su quello incrementale DDM raggiunge un tasso di mancate
rilevazioni del 40–45%, contro lo 0% del drift improvviso.

### 5.2 Le altre varianti come mappa dei limiti

[LU19] elenca le varianti nate da DDM, e ciascuna identifica per contrasto un
limite dell'originale:

| Variante | Stadio modificato | Limite di DDM che affronta |
|---|---|---|
| **EDDM** (Baena-García et al., 2006) | 3 | Sensibilità insufficiente sui drift graduali |
| **HDDM** (Frías-Blanco et al.) | 4 | Le soglie di DDM sono euristiche; HDDM usa la disuguaglianza di Hoeffding per definire la regione critica |
| **FW-DDM** | 1 | La finestra temporale convenzionale non gestisce bene il drift graduale; introduce una finestra fuzzy |
| **LLDD** | 3 e 4 | DDM produce un verdetto globale; LLDD lo scompone in decisioni locali sui nodi di un albero |
| **DELM** | — | Non cambia il rilevamento ma il learner di base |

Questa tabella è utile in tesi: mostra che i limiti di DDM sono **noti,
catalogati e affrontati** dalla letteratura, il che è più solido che elencarli
come osservazioni personali.

### 5.3 Limiti che discendono direttamente dalla formulazione

Non sono affermazioni delle fonti, ma conseguenze verificabili delle formule
riportate al §3, e come tali vanno presentate:

- **Richiede le etichette vere.** DDM è classificato da [SK17] fra i metodi
  *espliciti (supervisionati)*: senza `y_true` il tasso di errore non è
  calcolabile. In produzione, con label delay, il rilevamento arriva con il
  ritardo con cui arrivano le etichette.
- **Solo errore binario.** La modellazione bernoulliana di `s_t` presuppone un
  esito a due valori. Non è applicabile all'errore di un modello di regressione,
  che è un reale non limitato — è precisamente il caso coperto da Page-Hinkley.
- **Unilaterale.** Le due soglie confrontano `p_t + s_t` con un **minimo**
  storico: DDM reagisce solo alla crescita dell'errore. Un cambiamento di
  concetto che lasciasse invariata o migliorasse l'accuratezza non verrebbe
  rilevato. Dal punto di vista operativo è desiderabile; dal punto di vista
  della caratterizzazione del fenomeno è un limite.

---

## 6. Domande di verifica

Domande a cui questo documento permette di rispondere citando la fonte.

**A quale famiglia appartiene DDM?**
Controllo statistico di processo, secondo la tassonomia di [SK17], §2.1.2 —
distinta dall'analisi sequenziale (CUSUM, PHT) e dal windowing adattivo (ADWIN).

**Su quale presupposto teorico si fonda?**
Il modello PAC: in ambiente stazionario il tasso di errore di un learner non
deve crescere; un suo aumento significativo viola il modello ed è assunto come
conseguenza di drift ([SK17], [LU19]).

**Perché i coefficienti 2 e 3?**
Sono i limiti di controllo a 2σ e 3σ delle carte SPC, corrispondenti a circa il
95% e il 99% di confidenza sotto approssimazione normale della binomiale.
Coerenti con la collocazione SPC di DDM.

**Qual è il contributo storico di DDM?**
Essere stato il primo algoritmo a definire i due livelli, warning e drift
([LU19]) — e con essi il protocollo di costruzione anticipata di un nuovo
learner durante la fase di warning.

**Qual è il limite principale?**
I drift graduali lenti: è la motivazione dichiarata per cui è stato sviluppato
EDDM ([SK17]).

**Che finestra usa?**
Landmark, non scorrevole: punto iniziale fisso, punto finale che si estende a
ogni nuova istanza ([LU19]).

---

## 7. Riferimenti

**Paper primario, da recuperare via accesso istituzionale**
- Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004). *Learning with Drift
  Detection*. SBIA 2004, LNCS 3171, pp. 286–295. Springer.
  DOI 10.1007/978-3-540-28645-5_29

**Fonti secondarie effettivamente consultate per questo documento**
- Sethi, T. S., Kantardzic, M. (2017). *On the reliable detection of concept
  drift from streaming unlabeled data*. Expert Systems with Applications, 82,
  77–99. arXiv:1704.00023
- Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2019). *Learning under
  Concept Drift: A Review*. IEEE TKDE, 31(12), 2346–2363. arXiv:2004.05785

**Citati dalle fonti sopra, non consultati direttamente**
- Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavaldà, R.,
  Morales-Bueno, R. (2006). *Early Drift Detection Method*. ECML PKDD Workshop
  on Knowledge Discovery from Data Streams.
- Nishida, K., Yamauchi, K. (2007). *Detecting concept drift using statistical
  testing* (STEPD).
- Ross, G. J., Adams, N. M., Tasoulis, D. K., Hand, D. J. (2012).
  *Exponentially weighted moving average charts for detecting concept drift*.
- Haussler, D. (1990) — modello PAC, citato da [SK17] come fondamento teorico.

**Implementazione**
- Montiel, J., Halford, M., Mastelini, S. M., et al. (2021). *River: machine
  learning for streaming data in Python*. Journal of Machine Learning Research,
  22(110), 1–8. — classe `river.drift.binary.DDM`, versione 0.22.0
