# DDM — Drift Detection Method: fondamenti dal paper originale

Documento di riferimento scientifico su DDM, redatto a partire dal **paper
primario**, letto integralmente. Ogni affermazione riporta la sezione da cui
proviene.

Per la parte pratica — integrazione nel framework, parametri esposti, risultati
sperimentali ottenuti in questa tesi — vedere `docs/ddm_overview.md`, che è
complementare e non sostituisce questo documento.

---

## Fonte primaria

> **Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004).** *Learning with
> Drift Detection*. In: Bazzan, A.L.C., Labidi, S. (eds) Advances in Artificial
> Intelligence — SBIA 2004. Lecture Notes in Computer Science, vol. 3171,
> pp. 286–295. Springer, Berlin/Heidelberg.
> DOI [10.1007/978-3-540-28645-5_29](https://doi.org/10.1007/978-3-540-28645-5_29)

Copia in `Papers/Learning_with_Drift_Detection.pdf` (10 pagine).
Affiliazioni degli autori: LIACC — University of Porto; Fac. Economics,
University of Porto; University of Aveiro.

Fonti secondarie usate solo per l'inquadramento tassonomico e per la storia
successiva dell'algoritmo:

- **[SK17]** Sethi, T. S., Kantardzic, M. (2017). *On the reliable detection of
  concept drift from streaming unlabeled data*. Expert Systems with
  Applications, 82, 77–99. arXiv:1704.00023 — in `Papers/`
- **[LU19]** Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2019).
  *Learning under Concept Drift: A Review*. IEEE TKDE, 31(12), 2346–2363.
  arXiv:2004.05785 — in `Papers/`

---

## 1. L'idea centrale: il concetto di *contesto*

Questo è il punto che le sintesi di terza mano perdono quasi sempre, e che
invece organizza tutto il paper. Gli autori definiscono (§1):

> **contesto**: un insieme di esempi in cui la funzione che genera gli esempi è
> stazionaria.

Il flusso di dati è assunto composto **da una successione di contesti**. Le
transizioni fra contesti possono essere **graduali**, con passaggio morbido fra
le distribuzioni, o **brusche**, quando la distribuzione cambia rapidamente.

L'obiettivo dichiarato non è quindi "segnalare che qualcosa è cambiato", ma
**identificare i confini fra contesti**. Gli autori sono espliciti sulla
motivazione (§1): se si sanno identificare i contesti, si sa quale informazione
è ormai obsoleta, e si può riapprendere il modello **usando solo l'informazione
rilevante per il contesto attuale**.

Nelle conclusioni (§5) lo riformulano in modo ancora più diretto: dal punto di
vista pratico, ciò che il metodo fa è **scegliere l'insieme di addestramento più
appropriato** alla distribuzione di classe corrente.

> **Perché conta per questa tesi.** DDM non è nato come rilevatore di allarmi:
> è nato come **selettore del training set**. È esattamente l'uso che ne farà
> il capitolo sul retraining, ed è il motivo per cui è l'algoritmo giusto per
> l'Objective 4 della proposta.

---

## 2. Il fondamento statistico

Il paper (§3) assume che gli esempi arrivino uno alla volta, in coppie
`(x_i, y_i)`, nel quadro dell'apprendimento **online**: a ogni passo il modello
prende una decisione, e solo dopo l'ambiente fornisce il riscontro.

Per un insieme di esempi, l'errore è una variabile casuale proveniente da
**prove di Bernoulli**, e la distribuzione binomiale dà la forma generale della
probabilità per la variabile che rappresenta il numero di errori in un campione
di `n` esempi. Per ogni punto `i` della sequenza:

```
    p_i   = tasso di errore, cioe' la probabilita' di osservare False
    s_i   = sqrt( p_i * (1 - p_i) / i )
```

### La giustificazione teorica: una precisazione

Il paper afferma (§3) che la **teoria statistica garantisce** che, finché la
distribuzione di classe è stazionaria, il tasso di errore `p_i` diminuisce al
crescere di `i`; un aumento significativo suggerisce quindi un cambiamento nella
distribuzione di classe e l'inadeguatezza del modello attuale.

⚠️ **Il riferimento che gli autori citano a supporto è [9] = Tom Mitchell,
*Machine Learning*, McGraw Hill, 1997** — un manuale, non un risultato PAC
specifico.

Vale la pena saperlo, perché diverse fonti secondarie — fra cui [SK17] e
[LU19] — presentano DDM come basato sul "presupposto del modello PAC". È
un'interpretazione ragionevole e ormai standard nella letteratura, ma **il
paper originale non usa quella formulazione**: parla di teoria statistica in
senso generale e cita un manuale. In tesi conviene scrivere che il fondamento è
la decrescita attesa dell'errore in regime stazionario, notando che la
letteratura successiva l'ha ricondotta al modello PAC.

### Da dove vengono i coefficienti 2 e 3

Il paper lo deriva esplicitamente, ed è la parte che le sintesi omettono
sempre. Testualmente da §3: per valori sufficientemente grandi della dimensione
campionaria la binomiale è bene approssimata da una **normale** con la stessa
media e varianza; assumendo che la distribuzione di probabilità sia invariata
quando il contesto è statico, l'**intervallo di confidenza** `1 − α/2` per `p`
**con `n > 30` esempi** è approssimativamente `p_i ± α · s_i`, dove il
parametro `α` dipende dal livello di confidenza.

I due livelli sono quindi due intervalli di confidenza:

```
    livello di WARNING   ->  confidenza 95%  ->  alpha = 2
                             p_i + s_i  >=  p_min + 2 * s_min

    livello di DRIFT     ->  confidenza 99%  ->  alpha = 3
                             p_i + s_i  >=  p_min + 3 * s_min
```

Il paper precisa che questi sono i livelli **usati negli esperimenti descritti**,
non valori imposti dal metodo: `α` è un parametro.

> **Nota implementativa.** La condizione `n > 30` per la validità
> dell'approssimazione normale è, con ogni evidenza, l'origine del valore
> predefinito `warm_start = 30` di `river.drift.binary.DDM`. È utile saperlo
> per motivare la scelta di alzarlo a 200 negli esperimenti di questa tesi: non
> si sta contraddicendo il paper, si sta osservando che 30 è il minimo per la
> validità dell'approssimazione, non un valore ottimale per la stabilità della
> baseline.

### I due registri

Testualmente (§3): il metodo gestisce due registri durante l'addestramento,
`p_min` e `s_min`; ogni volta che un nuovo esempio `i` viene processato, tali
valori sono aggiornati **quando `p_i + s_i` è minore di `p_min + s_min`**.

---

## 3. Il protocollo di riapprendimento, nelle parole degli autori

È il cuore operativo del metodo, e il paper lo descrive in modo preciso (§3).

Il livello di warning serve a **definire la dimensione ottimale della finestra
di contesto**. Tale finestra conterrà gli esempi vecchi che appartengono già al
nuovo contesto, più un numero minimo di esempi del vecchio contesto.

Data una sequenza in cui l'errore del modello cresce raggiungendo:

- il **livello di warning** all'esempio `k_w`
- il **livello di drift** all'esempio `k_d`

allora si dichiara un **nuovo contesto a partire da `k_w`**, e si induce un nuovo
modello decisionale **usando solo gli esempi da `k_w` a `k_d`**.

### La gestione dei falsi allarmi

Il paper la affronta esplicitamente, ed è un dettaglio importante: può accadere
di osservare una crescita dell'errore che raggiunge il livello di warning e poi
**decresce**. Gli autori assumono che tali situazioni corrispondano a un **falso
allarme**, e in quel caso **il contesto non viene cambiato**.

Il warning non è quindi un allarme debole: è uno **stato reversibile**. Il
buffer accumulato viene semplicemente scartato se il drift non si conferma.

### Applicabilità

Il metodo può essere applicato con qualunque algoritmo di apprendimento (§3).
Può essere implementato **direttamente all'interno** di algoritmi online e
incrementali, oppure **come wrapper** attorno a learner batch. Nel framework di
questa tesi è realizzato come wrapper, coerentemente con la seconda modalità.

---

## 4. La validazione sperimentale originale

Utile da citare nel capitolo di validazione, per confrontare le proprie
condizioni sperimentali con quelle degli autori.

### Algoritmi di apprendimento

Tre, scelti per rappresentazioni deliberatamente diverse (§4): **Perceptron**
(lineare), **rete neurale** (combinazione non lineare di attributi), **albero di
decisione** (forma normale disgiuntiva). L'obiettivo dichiarato è mostrare
l'**indipendenza del metodo dall'algoritmo di apprendimento**.

### Gli otto dataset artificiali

Tutti a due classi, 50% di esempi per classe in ogni contesto, **1000 esempi
generati casualmente per contesto**, almeno due versioni del concetto target.

| Dataset | Tipo di drift | Caratteristica |
|---|---|---|
| SINE1 | brusco | senza rumore, 2 attributi rilevanti, `y = sin(x)` |
| SINE2 | brusco | `y < 0.5 + 0.3·sin(3πx)` |
| SINIRREL1 | brusco | SINE1 più 2 attributi irrilevanti |
| SINIRREL2 | brusco | SINE2 più 2 attributi irrilevanti |
| **CIRCLES** | **graduale** | senza rumore, quattro contesti definiti da quattro cerchi |
| GAUSS | brusco | **con rumore**, esempi normalmente distribuiti |
| STAGGER | brusco | attributi **simbolici**, senza rumore |
| MIXED | brusco | attributi booleani e numerici |

> **Osservazione da riportare in tesi.** Il paper originale **include un caso di
> drift graduale** (CIRCLES). L'affermazione, molto diffusa, secondo cui "DDM
> fallisce sui drift graduali" **non proviene da Gama et al.**: proviene dalla
> letteratura successiva, in particolare dagli autori di EDDM, riportata da
> [SK17]. È una precisazione che vale la pena fare, perché distingue chi ha
> letto il paper da chi ha letto le sintesi.

### Risultati sui dataset artificiali

Tasso di errore finale, con e senza rilevamento attivo (Tabella 1 del paper):

| Dataset | Perceptron | | Rete neurale | | Albero |  |
|---|---|---|---|---|---|---|
| | senza | **con** | senza | **con** | senza | **con** |
| STAGGER | 0,048 | **0,029** | 0,351 | **0,002** | 0,265 | **0,016** |
| SINE1 | 0,126 | **0,115** | 0,489 | **0,019** | 0,490 | **0,081** |
| SINIRREL1 | 0,159 | **0,139** | 0,479 | **0,068** | 0,483 | **0,088** |
| SINE2 | 0,271 | **0,262** | 0,492 | **0,118** | 0,477 | **0,100** |
| SINIRREL2 | 0,281 | 0,281 | 0,477 | **0,059** | 0,485 | **0,084** |
| MIXED | 0,100 | 0,111 | 0,240 | **0,065** | 0,491 | **0,465** |
| GAUSS | 0,384 | 0,386 | 0,395 | **0,150** | 0,380 | **0,144** |
| CIRCLES | 0,410 | 0,413 | 0,233 | **0,225** | 0,205 | **0,109** |

Gli autori osservano (§4.2) che il metodo è efficace con tutti gli algoritmi,
ma che **le differenze sono più significative con la rete neurale e con
l'albero di decisione**. Nelle conclusioni lo formulano come regola: il metodo
è più efficace con algoritmi dotati di **maggiore capacità di rappresentare
generalizzazioni**. Sul Perceptron, infatti, in tre casi su otto il rilevamento
peggiora leggermente il risultato.

### ELEC2 — lo stesso dataset usato in questa tesi

Il paper dedica a ELEC2 la §4.3, e le informazioni sono direttamente rilevanti.

Il dataset proviene dal mercato elettrico australiano del New South Wales, dove
i prezzi non sono fissi e sono determinati da domanda e offerta, aggiornati ogni
cinque minuti. Contiene **45.312 istanze dal 7 maggio 1996 al 5 dicembre 1998**,
ciascuna riferita a un periodo di 30 minuti, quindi **48 istanze per giorno**.
L'etichetta di classe identifica la variazione del prezzo rispetto a una media
mobile delle ultime 24 ore, il che rimuove l'effetto delle tendenze di lungo
periodo.

Gli autori riportano inoltre, citando Harries, due fatti sul dataset: la
**stagionalità** della formazione del prezzo e la sensibilità a eventi di breve
termine come le fluttuazioni meteorologiche; e il fatto che durante il periodo
osservato il mercato **fu ampliato con l'inclusione di aree adiacenti**,
consentendo di vendere l'eccesso di produzione di una regione in quella
adiacente, con effetto di attenuazione dei prezzi estremi.

> ⚠️ **Discrepanza importante da dichiarare in tesi.** Il paper descrive ELEC2
> come composto da **5 campi** più l'etichetta: giorno della settimana, marca
> temporale, domanda elettrica NSW, domanda elettrica Vic, trasferimento
> programmato fra stati. Il file usato in questa tesi,
> `electricity-normalized.csv`, ha invece **8 feature** (`date`, `day`,
> `period`, `nswprice`, `nswdemand`, `vicprice`, `vicdemand`, `transfer`) più la
> classe. Si tratta della versione normalizzata ed estesa diffusa tramite
> OpenML/MOA, non di quella descritta dagli autori. La differenza va dichiarata:
> i risultati non sono direttamente confrontabili con quelli del paper.
>
> Va inoltre ricordato che l'ampliamento del mercato alle aree adiacenti,
> menzionato dagli autori, è verosimilmente la causa della **regione costante
> nelle feature di Victoria** individuata nell'analisi di qualità del dato di
> questo lavoro: i dati di Victoria non esistevano prima di quell'ampliamento.
> Il paper originale fornisce quindi la spiegazione storica dell'artefatto.

Sui risultati, gli autori sono cauti in un modo che vale la pena imitare:
**non sanno se e quando il drift avvenga**, e lo dichiarano. Costruiscono quindi
un limite inferiore per confronto, cercando esaustivamente il segmento di
training con la migliore prestazione sul test set — operazione che, come
riconoscono, **non è praticabile nella realtà** perché guarda le etichette del
test set.

| Test set | Limite inferiore | Tutti i dati | Ultimo anno | **Drift Detection** |
|---|---|---|---|---|
| Ultimo giorno | 0,104 | 0,187 | 0,125 | **0,104** |
| Ultima settimana | 0,190 | 0,235 | 0,247 | **0,199** |

Con il test a un giorno il metodo **raggiunge esattamente il limite inferiore**;
con quello a una settimana gli si avvicina molto. Il modello viene costruito
sugli ultimi 3.836 esempi nel primo caso e sui 3.548 più recenti nel secondo.

### Il controllo negativo su ADULT

Gli autori eseguono una verifica che merita attenzione (§4.3, chiusura). Testano
il metodo sul dataset **ADULT**, costruito da dati censuari raccolti in un
singolo istante, in cui **il concetto dovrebbe essere stabile**. Usando un
albero di decisione, **il metodo non rileva mai drift**. Gli autori lo
qualificano come aspetto importante, perché costituisce evidenza che il metodo è
**robusto ai falsi allarmi**.

> **Da valorizzare in tesi.** È esattamente la stessa logica del controllo
> negativo costruito in questo lavoro sui dataset Friedman, dove le feature sono
> uniformi per costruzione e ogni segnalazione è per definizione un falso
> positivo. Poter scrivere che il protocollo di validazione adottato replica
> quello degli autori originali del metodo è un argomento forte.

---

## 5. Limiti

### 5.1 Riconosciuti dagli autori

- **Efficacia dipendente dal learner** (§5): il metodo è più efficace con
  algoritmi dotati di maggiore capacità di generalizzazione. Sul Perceptron il
  guadagno è marginale o assente.
- **Verifica su ELEC2 non conclusiva** (§4.3): gli autori dichiarano di non
  sapere se e quando il drift avvenga sui dati reali.

### 5.2 Attribuiti dalla letteratura successiva

- **Drift graduali lenti.** Secondo [SK17], EDDM (Baena-García et al., 2006) è
  stato sviluppato come estensione di DDM proprio per essere adatto ai drift
  graduali lenti, «dove DDM in precedenza falliva». EDDM monitora la **distanza
  in numero di campioni fra due errori consecutivi** anziché il tasso di errore.
  Come notato al §4, **questa non è un'affermazione del paper originale**.
- **Soglie euristiche.** [LU19] riporta che HDDM modifica il quarto stadio di
  DDM usando la disuguaglianza di Hoeffding per identificare la regione critica.
- **Verdetto globale.** [LU19]: LLDD scompone il problema in decisioni locali
  sui nodi di un albero.

### 5.3 Che discendono dalla formulazione

- **Richiede le etichette vere.** Il tasso di errore non è calcolabile senza
  `y_true`. In produzione, con label delay, il rilevamento arriva con il ritardo
  con cui arrivano le etichette.
- **Unilaterale.** Entrambe le soglie confrontano `p_i + s_i` con un **minimo**
  storico: DDM reagisce alla sola crescita dell'errore.
- **Finestra landmark, non scorrevole.** Secondo la ricostruzione di [LU19], il
  punto iniziale è fisso e quello finale si estende a ogni nuova istanza. È
  coerente con il meccanismo di dichiarazione di un nuovo contesto descritto dal
  paper: dopo un drift si riparte.

### 5.4 Una correzione importante sulla regressione

Nelle conclusioni (§5) gli autori scrivono che l'algoritmo **potrebbe essere
applicato con qualunque funzione di perdita**, dati valori appropriati di `α`, e
che risultati preliminari **in dominio di regressione** usando l'errore
quadratico medio confermano quelli presentati.

⚠️ Va quindi corretta un'affermazione troppo netta presente altrove in questa
documentazione. La formulazione corretta è:

- **il metodo DDM, come concepito dagli autori, non è limitato alla
  classificazione**; gli autori ne prospettano esplicitamente l'uso con altre
  funzioni di perdita;
- **la sua realizzazione in `river.drift.binary.DDM` sì**, perché implementa la
  derivazione bernoulliana `s_i = sqrt(p_i(1−p_i)/i)`, che presuppone un esito a
  due valori.

Il vincolo che il framework di questa tesi impone — `DDMStrategy` rifiuta valori
non binari — è quindi corretto **rispetto all'implementazione usata**, ma non va
presentato come un limite del metodo. La generalizzazione a una funzione di
perdita continua richiederebbe una diversa stima della varianza ed è una
direzione di sviluppo futuro suggerita dagli autori stessi.

---

## 6. Domande di verifica

**Qual è l'idea centrale del paper?**
Il concetto di **contesto**: un insieme di esempi in cui la funzione generatrice
è stazionaria. Il flusso è una successione di contesti, e il metodo serve a
individuarne i confini per riapprendere solo sull'informazione rilevante.

**Da dove vengono i coefficienti 2 e 3?**
Sono i moltiplicatori `α` dell'intervallo di confidenza `p_i ± α·s_i`, valido
per `n > 30` sotto approssimazione normale della binomiale. Corrispondono al 95%
e al 99% di confidenza, e il paper li dichiara come valori usati negli
esperimenti, non come costanti del metodo.

**Su quali dati si riaddestra dopo un drift?**
Sugli esempi da `k_w` a `k_d`, cioè dal livello di warning al livello di drift.
Il warning definisce la dimensione della finestra di contesto.

**Cosa succede se il warning non è seguito da un drift?**
È trattato come falso allarme e il contesto non cambia. Il warning è uno stato
reversibile.

**È vero che DDM fallisce sui drift graduali?**
È l'affermazione degli autori di EDDM, riportata da [SK17]. Il paper originale
include un dataset a drift graduale (CIRCLES) e non dichiara quel limite.

**DDM funziona sulla regressione?**
Il metodo sì, secondo gli autori, con una funzione di perdita appropriata e
risultati preliminari con MSE. L'implementazione di river no, perché usa la
derivazione bernoulliana.

**Gli autori hanno verificato i falsi allarmi?**
Sì, sul dataset ADULT, a concetto stabile: il metodo non rileva mai drift.

**Che risultati hanno ottenuto su ELEC2?**
Errore 0,104 sul test a un giorno, pari al limite inferiore ottenuto per ricerca
esaustiva, e 0,199 sul test a una settimana contro un limite inferiore di 0,190.

---

## 7. Riferimenti

**Fonte primaria**
- Gama, J., Medas, P., Castillo, G., Rodrigues, P. (2004). *Learning with Drift
  Detection*. SBIA 2004, LNCS 3171, pp. 286–295. Springer.
  DOI 10.1007/978-3-540-28645-5_29

**Citati dal paper originale e rilevanti per questa tesi**
- Mitchell, T. (1997). *Machine Learning*. McGraw Hill. — rif. [9] del paper,
  supporto della garanzia statistica sulla decrescita dell'errore
- Harries, M. (1999). *Splice-2 comparative evaluation: Electricity pricing*.
  Technical report, The University of South Wales. — rif. [2], fonte di ELEC2
- Widmer, G., Kubat, M. (1996). *Learning in the presence of concept drift and
  hidden contexts*. Machine Learning, 23, 69–101. — rif. [11], famiglia FLORA
- Klinkenberg, R., Joachims, T. (2000). *Detecting concept drift with support
  vector machines*. ICML-00, pp. 487–494. — rif. [5]
- Kubat, M., Widmer, G. (1995). *Adapting to drift in continuous domain*. ECML,
  pp. 307–310. — rif. [6], origine dei dataset artificiali
- Blake, C., Keogh, E., Merz, C. J. (1999). *UCI repository of Machine Learning
  databases*. — rif. [1], fonte del dataset ADULT

**Fonti secondarie, per la storia successiva dell'algoritmo**
- Sethi, T. S., Kantardzic, M. (2017). Expert Systems with Applications, 82,
  77–99. arXiv:1704.00023
- Lu, J., et al. (2019). IEEE TKDE, 31(12), 2346–2363. arXiv:2004.05785
- Baena-García, M., et al. (2006). *Early Drift Detection Method*. ECML PKDD
  Workshop on Knowledge Discovery from Data Streams. — citato da [SK17], non
  consultato direttamente

**Implementazione**
- Montiel, J., et al. (2021). *River: machine learning for streaming data in
  Python*. JMLR, 22(110), 1–8. — classe `river.drift.binary.DDM`, v. 0.22.0
