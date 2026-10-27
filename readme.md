# Serie A Match Predictor 

PyTorch-based neural network for predicting Serie A match outcomes using historical data.

## Key Features
* **Anti-Data Leakage:** Utilizzo esclusivo di medie mobili (Rolling Averages) e storico H2H calcolati pre-match.
* **Gap Stats:** Feature engineering avanzato sui differenziali di rendimento (xG, tiri, possesso) per far percepire al modello l'equilibrio effettivo in campo.
* **Class Weights Manuali:** Bilanciamento della CrossEntropyLoss `[1.0, 1.2, 1.1]` per contrastare lo sbilanciamento statistico e intercettare fisiologicamente i pareggi (X).
* **Auto-Tuning:** Ricerca automatizzata degli iperparametri (hidden layers, learning rate, ecc.) tramite Optuna con TimeSeriesSplit.

## Struttura del Progetto
* `retefinale.py`: Pipeline completa di addestramento e ottimizzazione.
* `pronostici.py`: Motore di inferenza rapido per generare previsioni in tempo reale su nuove partite.
* `serie_a_predictor.pt`: Pesi del modello addestrato.
* `complete_data/`: Directory contenente il dataset storico Excel.

## Installazione e Uso
1. Clona il repository
2. Installa le librerie necessarie:
   ```bash
   pip install -r requirements.txt
3. aggiorna le squadre dal blocco in basso del file pronostici.py
4. esegui pronostici.py
