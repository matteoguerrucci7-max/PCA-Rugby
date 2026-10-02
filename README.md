# PCA-Rugby
# Stili di gioco e gruppi di lavoro bilanciati (PCA + metodo delle potenze)

Progetto di Analisi Numerica. Una rosa di rugby di 25 giocatori è descritta da 10 statistiche per 80 minuti.

**Domande**
1. Quanti "assi di stile di gioco" bastano a descrivere i giocatori, e quali sono?
2. Come si divide la squadra in gruppi di lavoro equilibrati, in cui chi è forte su un asse lavora con chi è debole?

> I dati sono **sintetici** (`synthetic_data.py`): profili medi per ruolo più rumore.

## Metodo

| Passo | Cosa | Strumento numerico |
|---|---|---|
| 1 | Standardizzazione, matrice di covarianza C = XᵀX/(N−1) | — |
| 2 | Componenti principali: max bᵀCb con ‖b‖=1 ⇒ Cb = λb | lagrangiana |
| 3 | Calcolo delle componenti | **metodo delle potenze** + Rayleigh + **deflazione**, verificato con la **SVD** |
| 4 | Gruppi bilanciati: min SS_tra (⇔ max SS_dentro) | decomposizione della varianza + scambi a coppie |
| 5 | Direzioni di lavoro: asse debole rispetto al proprio ruolo, statistiche da allenare, mentore | — |
| 6 | Accelerazione: shift, potenze inverse con shift | fattorizzazione **LU** una volta, poi sostituzioni |
| 7 | Affidabilità degli assi: rotazione degli autovettori con dati perturbati | condizionamento e gap tra autovalori |
| 8 | Approssimazione di rango k e dati mancanti | **Eckart–Young**, imputazione con SVD iterativa (**punto fisso**) |

La derivazione completa è in [`teoria.pdf`](teoria.pdf). Riferimento teorico: Deisenroth, Faisal, Ong, *Mathematics for Machine Learning* (2020), cap. 6 e 10.
🎬 [Guarda l'animazione](https://matteoguerrucci7-max.github.io/PCA-Rugby/animazione/)


## Risultati principali

- **Assi**: b₁ = contatto ↔ gioco al largo, b₂ = portatore di palla ↔ chi porta poco palla. Le prime 3 componenti spiegano il 79,6% della varianza.
- **Metodo delle potenze**: autovalori uguali a quelli della SVD; fattore di convergenza misurato 0,760 contro |λ₂/λ₁| = 0,754 teorico. Dopo la deflazione b₂ converge in 16 iterazioni (λ₃/λ₂ = 0,35), b₁ in 58 (λ₂/λ₁ = 0,75).
- **Gruppi**: SS_tra = 0,30 su SS_tot = 191,1 (0,16%). Con 5000 divisioni casuali: media 31,7, migliore 4,2.
- **Shift**: con σ* = (λ₂+λ_D)/2 le iterazioni per b₁ scendono da 47 a 29; con le potenze inverse (σ vicino a λⱼ) bastano 5–9 iterazioni.
- **Condizionamento**: potenze e potenze inverse (σ = 0) danno K(C) = λ_max/λ_min = 108,4 = K(X)², con K(X) = 10,4 dalla SVD.
- **Affidabilità**: con il 10% di rumore b₁ e b₂ ruotano di circa 2,5–3°, b₄ di 8° (gap 0,18): si interpretano solo i primi due assi.
- **Dati mancanti**: l'imputazione di rango 2 riduce l'errore del 37% rispetto alla media; con rango troppo alto il punto fisso non converge.

| Convergenza delle potenze | Gruppi nello spazio degli stili | Contro divisioni casuali |
| --- | --- | --- |
| ![](figure/fig_potenze_convergenza.png) | ![](figure/fig_gruppi.png) | ![](figure/fig_confronto_casuali.png) |

| Rango k | Dati mancanti | Sensibilità |
| --- | --- | --- |
| ![](figure/fig_eckart_young.png) | ![](figure/fig_imputazione.png) | ![](figure/fig_sensibilita.png) |
## Esempio: direzioni di lavoro

Per ogni giocatore si confrontano le coordinate sugli assi con la media del
**proprio ruolo** (la PCA descrive stili, non livelli). Se uno è
sotto la media del ruolo su un asse, il programma indica le statistiche
da allenare e un mentore, cioè un giocatore forte su quell'asse.

| Giocatore         | Asse debole | Da allenare                                          | Mentore         |
| ----------------- | ----------- | ---------------------------------------------------- | --------------- |
| Pilone 3          | b2 (−0,28)  | ruck_offensive                                       | Seconda linea 1 |
| Seconda linea 1   | b1 (−0,46)  | ruck_offensive, placcaggi_effettuati, placcaggi_dominanti | Terza linea 2 |
| Terza linea 4     | b2 (−0,43)  | portate, offload, metri_guadagnati                   | Estremo 1       |
| Apertura 1        | b2 (−0,29)  | passaggi, calci_in_gioco                             | Centro 2        |
| Centro 3          | b2 (−0,56)  | portate, offload, difensori_battuti                  | Estremo 1       |

Gli altri 20 giocatori sono in linea o sopra la media del loro ruolo.
Output completo: `python blocco3_gruppi.py`.

## Uso

```bash
pip install -r requirements.txt
python blocco1_pca_svd.py      # covarianza, SVD, varianza spiegata, pesi degli assi
python blocco2_potenze.py      # metodo delle potenze + deflazione, grafico di convergenza
python blocco3_gruppi.py       # gruppi bilanciati, confronto con il caso, direzioni di lavoro
python blocco4_rango_imputazione.py   # Eckart-Young, imputazione dei dati mancanti
python blocco5_shift_sensibilita.py   # shift, potenze inverse (LU), sensibilita' al rumore
```

## Limiti

- Dati sintetici e N = 25: le componenti oltre la seconda sono poco affidabili (λ₃ e λ₄ sono vicini).
- L'algoritmo di scambio trova un **minimo locale**, non garantisce l'ottimo globale (30 partenze casuali).
- La PCA descrive **stili, non livelli**: per le direzioni di lavoro ogni giocatore è confrontato con la media del suo ruolo.

## Sviluppi

Dati reali di un campionato intero: assi stimati su centinaia di giocatori, gruppi formati dentro una squadra. Per N grande si calcola Cv = Xᵀ(Xv)/(N−1) senza formare C.
