"""
Serie A Match Predictor - Training Pipeline by Pier Manca
Script completo per l'addestramento end-to-end di una rete neurale in PyTorch.
Include Time Series Cross-Validation, feature engineering (Gap Stats, H2H) 
e hyperparameter tuning automatizzato con Optuna.
"""
#%%
# --- 1. Importazioni e Setup Iniziale ---
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import optuna

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in uso: {device}")

#%%
# --- 2. Caricamento dei dati e Ordinamento Temporale delle partite---
dataset = pd.read_excel('complete_data/completo.xlsx')
dataset['date'] = pd.to_datetime(dataset['date'])
dataset = dataset.sort_values('date').reset_index(drop=True)

#%%
# --- 3. Creazione del Target (0 = 1, 1 = X, 2 = 2) per gli esiti ---
condizioni = [
    (dataset['home_goals'] > dataset['away_goals']),
    (dataset['home_goals'] == dataset['away_goals']),
    (dataset['home_goals'] < dataset['away_goals'])
]
dataset['target'] = np.select(condizioni, [0, 1, 2])

#%%
# --- 4. Creazione dei parametri (Medie Mobili, H2H e GAP STATS) ---
home_stats = ['HS', 'HST', 'HC', 'HF', 'HY', 'HR', 'home_xg', 'home_np_xg', 'home_np_xg_difference', 'home_ppda', 'home_deep_completions']
away_stats = ['AS', 'AST', 'AC', 'AF', 'AY', 'AR', 'away_xg', 'away_np_xg', 'away_np_xg_difference', 'away_ppda', 'away_deep_completions']

# 4a. Statistiche generali delle ultime 5 partite
for col in home_stats:
    dataset[f'{col}_last5_home_avg'] = dataset.groupby('home_team')[col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

for col in away_stats:
    dataset[f'{col}_last5_away_avg'] = dataset.groupby('away_team')[col].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )

# 4b. Storico Scontri Diretti (H2H)
dataset['h2h_home_goals_avg'] = dataset.groupby(['home_team', 'away_team'])['home_goals'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
).fillna(0) 

dataset['h2h_away_goals_avg'] = dataset.groupby(['home_team', 'away_team'])['away_goals'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
).fillna(0)

# 4c. GAP STATS (La differenza assoluta tra le due squadre)
dataset['gap_xg'] = abs(dataset['home_xg_last5_home_avg'] - dataset['away_xg_last5_away_avg'])
dataset['gap_shots'] = abs(dataset['HS_last5_home_avg'] - dataset['AS_last5_away_avg'])
dataset['gap_shots_target'] = abs(dataset['HST_last5_home_avg'] - dataset['AST_last5_away_avg'])
dataset['gap_corners'] = abs(dataset['HC_last5_home_avg'] - dataset['AC_last5_away_avg'])
dataset['gap_fouls'] = abs(dataset['HF_last5_home_avg'] - dataset['AF_last5_away_avg'])
dataset['gap_yellows'] = abs(dataset['HY_last5_home_avg'] - dataset['AY_last5_away_avg'])
dataset['gap_ppda'] = abs(dataset['home_ppda_last5_home_avg'] - dataset['away_ppda_last5_away_avg'])

# 4d. Eliminazione totale di dati a posteriori per evitare il Data Leakage
colonne_post_match = ['home_goals', 'away_goals', 'home_expected_points', 'away_expected_points'] + home_stats + away_stats
dataset_pulito = dataset.drop(columns=colonne_post_match).dropna().reset_index(drop=True)

dataset_origin = dataset_pulito.copy()

#%%
# --- 5. Codifica Numerica delle Squadre ---
tutte_le_squadre = pd.concat([dataset_pulito['home_team'], dataset_pulito['away_team']]).astype(str).unique()
encoder = LabelEncoder()
encoder.fit(tutte_le_squadre)

dataset_pulito['home_team'] = encoder.transform(dataset_pulito['home_team'].astype(str))
dataset_pulito['away_team'] = encoder.transform(dataset_pulito['away_team'].astype(str))

#%%
# --- 6. Creazione Tensori per il Modello ---
X = dataset_pulito.drop(columns=['game', 'date', 'target']).values.astype(np.float32)
y = dataset_pulito['target'].values.astype(np.int64)

input_features = X.shape[1]

#%%
# --- 7. Isolamento Test Set Finale (15%) ---
totale_partite = len(X)
test_size = int(totale_partite * 0.15)
train_val_size = totale_partite - test_size

X_train_val, y_train_val = X[:train_val_size], y[:train_val_size]
X_test, y_test = X[train_val_size:], y[train_val_size:]

#%%
# --- 8. Time Series Cross-Validation Setup ---
K_SPLITS = 3
tscv = TimeSeriesSplit(n_splits=K_SPLITS)

def get_time_series_loaders(X_tv, y_tv, split_index, batch_size):
    train_index, val_index = list(tscv.split(X_tv))[split_index]
    
    X_train, y_train = X_tv[train_index], y_tv[train_index]
    X_val, y_val = X_tv[val_index], y_tv[val_index]
    
    low = X_train.min(axis=0)
    high = X_train.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    
    X_train = ((X_train - low) / span).astype(np.float32)
    X_val = ((X_val - low) / span).astype(np.float32)
    
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), 
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), 
                            batch_size=256, shuffle=False)
    
    return train_loader, val_loader, y_train

#%%
# --- 9. Architettura della Rete Neurale ---
class SerieAPredictor(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_size, dropout_rate=0.0):
        super().__init__()
        layers, width = [], in_features
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_size), nn.ReLU()])
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            width = hidden_size
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def make_model(hidden_layers, hidden_size, dropout_rate):
    torch.manual_seed(SEED)
    return SerieAPredictor(input_features, hidden_layers, hidden_size, dropout_rate).to(device)

#%%
# --- 10. Funzione Obiettivo per Optuna (PESI MANUALI BILANCIATI) ---
def objective(trial):
    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    scores = []
    epochs = 40 
    
    for split_idx in range(K_SPLITS):
        train_loader, val_loader, _ = get_time_series_loaders(X_train_val, y_train_val, split_idx, batch_size)
        
        # --- CLASS WEIGHTS MANUALI: 1.0 (Casa), 1.2 (Pareggio), 1.1 (Trasferta) ---
        pesi_tensor = torch.tensor([1.0, 1.2, 1.1], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=pesi_tensor)
        
        model = make_model(hidden_layers, hidden_size, dropout_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_logits = model(x_val)
                    preds.extend(val_logits.argmax(dim=1).cpu().numpy())
                    targets.extend(y_val.cpu().numpy())
            
            val_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
        
        scores.append(best_val_f1)
        trial.report(np.mean(scores), split_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(scores)

#%%
# --- 11. Esecuzione Optuna per la ricerca degli iperparametri migliori ---
print("Ricerca iperparametri in corso...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED), 
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
study.optimize(objective, n_trials=20)

print(f"\nMiglior F1 Score (Validation): {study.best_value:.4f}")
print("Migliori parametri trovati:", study.best_params)

#%%
# --- 12. Addestramento del Modello Finale ---
best_params = study.best_params

low_final = X_train_val.min(axis=0)
high_final = X_train_val.max(axis=0)
span_final = np.where(high_final > low_final, high_final - low_final, 1.0)

X_train_val_scaled = ((X_train_val - low_final) / span_final).astype(np.float32)
X_test_scaled = ((X_test - low_final) / span_final).astype(np.float32)

final_train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_val_scaled), torch.from_numpy(y_train_val)), 
                                batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_scaled), torch.from_numpy(y_test)), 
                         batch_size=256, shuffle=False)

final_model = make_model(best_params['hidden_layers'], best_params['hidden_size'], best_params['dropout_rate'])
optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])

# --- CLASS WEIGHTS FINALI MANUALI ---
pesi_tensor_final = torch.tensor([1.0, 1.2, 1.1], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=pesi_tensor_final)

final_model.train()
for epoch in range(40):
    for x_batch, y_batch in final_train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(final_model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

#%%
# --- 13. Valutazione sul Test Set e Matrice di Confusione ---
final_model.eval()
test_preds, test_targets = [], []

with torch.no_grad():
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch = x_test_batch.to(device)
        logits = final_model(x_test_batch)
        test_preds.extend(logits.argmax(dim=1).cpu().numpy())
        test_targets.extend(y_test_batch.numpy())

print("\n--- REPORT SUL TEST SET FINALE ---")
class_names = ['1 (Casa)', 'X (Pareggio)', '2 (Trasferta)']
print(classification_report(test_targets, test_preds, target_names=class_names, zero_division=0))

cm = confusion_matrix(test_targets, test_preds, normalize='true')
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice di Confusione (Test Set)')
plt.ylabel('Risultato Reale')
plt.xlabel('Previsione')
plt.show()

torch.save(final_model.state_dict(), 'serie_a_predictor.pt')
print("Modello salvato: 'serie_a_predictor.pt'")

#%%
# --- 14. Funzione di Previsione Rapida  ---
def prevedi_partita_test(home_team_name, away_team_name):
    final_model.eval()
    try:
        home_id = encoder.transform([home_team_name])[0]
        away_id = encoder.transform([away_team_name])[0]
    except ValueError:
        print(f"Errore: Squadra non trovata. Ecco le squadre valide:")
        print(sorted(dataset['home_team'].unique()))
        return

    feature_columns = dataset_origin.drop(columns=['game', 'date', 'target']).columns
    
    home_history = dataset_origin[dataset_origin['home_team'] == home_team_name].iloc[-1]
    away_history = dataset_origin[dataset_origin['away_team'] == away_team_name].iloc[-1]
    
    matchup_passati = dataset[(dataset['home_team'] == home_team_name) & (dataset['away_team'] == away_team_name)]
    if len(matchup_passati) > 0:
        valore_h2h_home = matchup_passati['home_goals'].tail(5).mean()
        valore_h2h_away = matchup_passati['away_goals'].tail(5).mean()
    else:
        valore_h2h_home, valore_h2h_away = 0.0, 0.0

    match_features = {}
    match_features['home_team'] = home_id
    match_features['away_team'] = away_id
    match_features['h2h_home_goals_avg'] = valore_h2h_home
    match_features['h2h_away_goals_avg'] = valore_h2h_away
    
    for col in feature_columns:
        if 'home_avg' in col and 'gap' not in col:
            match_features[col] = home_history[col]
        elif 'away_avg' in col and 'gap' not in col:
            match_features[col] = away_history[col]

    match_features['gap_xg'] = abs(match_features.get('home_xg_last5_home_avg', 0) - match_features.get('away_xg_last5_away_avg', 0))
    match_features['gap_shots'] = abs(match_features.get('HS_last5_home_avg', 0) - match_features.get('AS_last5_away_avg', 0))
    match_features['gap_shots_target'] = abs(match_features.get('HST_last5_home_avg', 0) - match_features.get('AST_last5_away_avg', 0))
    match_features['gap_corners'] = abs(match_features.get('HC_last5_home_avg', 0) - match_features.get('AC_last5_away_avg', 0))
    match_features['gap_fouls'] = abs(match_features.get('HF_last5_home_avg', 0) - match_features.get('AF_last5_away_avg', 0))
    match_features['gap_yellows'] = abs(match_features.get('HY_last5_home_avg', 0) - match_features.get('AY_last5_away_avg', 0))
    match_features['gap_ppda'] = abs(match_features.get('home_ppda_last5_home_avg', 0) - match_features.get('away_ppda_last5_away_avg', 0))
    
    input_vector = [match_features[col] for col in feature_columns]
                
    input_scaled = (np.array(input_vector, dtype=np.float32) - low_final) / span_final
    input_tensor = torch.tensor(input_scaled).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probabilita = F.softmax(final_model(input_tensor), dim=1).cpu().numpy()[0]
        
    print(f"\n--- TEST PREVISIONE: {home_team_name} vs {away_team_name} ---")
    print(f"1: {probabilita[0] * 100:.1f}% | X: {probabilita[1] * 100:.1f}% | 2: {probabilita[2] * 100:.1f}%")
    print(f"Pronostico: {['1', 'X', '2'][np.argmax(probabilita)]}")

#%%
# --- 15. Esempio di test rapidi post-addestramento ---
prevedi_partita_test('Milan', 'Inter')

# %%
