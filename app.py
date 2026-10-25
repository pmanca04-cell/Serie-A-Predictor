import streamlit as st
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# --- SETUP INTERFACCIA ---
st.set_page_config(page_title="Serie A Predictor", page_icon="⚽")
st.title("⚽ Predittore Serie A")
st.markdown("Previsioni basate su Reti Neurali e Gap Stats")

# --- PARAMETRI RETE ---
BEST_HIDDEN_LAYERS = 1  
BEST_HIDDEN_SIZE = 128   
FILE_PESI = 'serie_a_predictor.pt'
FILE_DATI = 'complete_data/completo.xlsx'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SerieAPredictor(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_size):
        super().__init__()
        layers, width = [], in_features
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_size), nn.ReLU()])
            width = hidden_size
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# @st.cache_resource fa in modo che la rete venga caricata una sola volta, velocizzando l'app
@st.cache_resource 
def carica_modello_e_dati():
    dataset = pd.read_excel(FILE_DATI)
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset = dataset.sort_values('date').reset_index(drop=True)

    home_stats = ['HS', 'HST', 'HC', 'HF', 'HY', 'HR', 'home_xg', 'home_np_xg', 'home_np_xg_difference', 'home_ppda', 'home_deep_completions']
    away_stats = ['AS', 'AST', 'AC', 'AF', 'AY', 'AR', 'away_xg', 'away_np_xg', 'away_np_xg_difference', 'away_ppda', 'away_deep_completions']

    for col in home_stats:
        dataset[f'{col}_last5_home_avg'] = dataset.groupby('home_team')[col].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    for col in away_stats:
        dataset[f'{col}_last5_away_avg'] = dataset.groupby('away_team')[col].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

    dataset['h2h_home_goals_avg'] = dataset.groupby(['home_team', 'away_team'])['home_goals'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0) 
    dataset['h2h_away_goals_avg'] = dataset.groupby(['home_team', 'away_team'])['away_goals'].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()).fillna(0)

    dataset['gap_xg'] = abs(dataset['home_xg_last5_home_avg'] - dataset['away_xg_last5_away_avg'])
    dataset['gap_shots'] = abs(dataset['HS_last5_home_avg'] - dataset['AS_last5_away_avg'])
    dataset['gap_shots_target'] = abs(dataset['HST_last5_home_avg'] - dataset['AST_last5_away_avg'])
    dataset['gap_corners'] = abs(dataset['HC_last5_home_avg'] - dataset['AC_last5_away_avg'])
    dataset['gap_fouls'] = abs(dataset['HF_last5_home_avg'] - dataset['AF_last5_away_avg'])
    dataset['gap_yellows'] = abs(dataset['HY_last5_home_avg'] - dataset['AY_last5_away_avg'])
    dataset['gap_ppda'] = abs(dataset['home_ppda_last5_home_avg'] - dataset['away_ppda_last5_away_avg'])

    colonne_post_match = ['home_goals', 'away_goals', 'home_expected_points', 'away_expected_points'] + home_stats + away_stats
    dataset_pulito = dataset.drop(columns=colonne_post_match).dropna().reset_index(drop=True)

    encoder = LabelEncoder()
    tutte_le_squadre = pd.concat([dataset_pulito['home_team'], dataset_pulito['away_team']]).astype(str).unique()
    encoder.fit(tutte_le_squadre)
    
    dataset_encoded = dataset_pulito.copy()
    dataset_encoded['home_team'] = encoder.transform(dataset_encoded['home_team'].astype(str))
    dataset_encoded['away_team'] = encoder.transform(dataset_encoded['away_team'].astype(str))
    
    cols_to_drop = ['game', 'date']
    if 'target' in dataset_encoded.columns:
        cols_to_drop.append('target')
        
    X_full = dataset_encoded.drop(columns=cols_to_drop).values.astype(np.float32)
    train_size = int(len(X_full) * 0.85)
    X_train = X_full[:train_size]
    
    low_final = X_train.min(axis=0)
    high_final = X_train.max(axis=0)
    span_final = np.where(high_final > low_final, high_final - low_final, 1.0)
    
    in_features = X_full.shape[1]
    
    modello = SerieAPredictor(in_features, BEST_HIDDEN_LAYERS, BEST_HIDDEN_SIZE).to(device)
    modello.load_state_dict(torch.load(FILE_PESI, map_location=device, weights_only=True), strict=False)
    modello.eval()
    
    return modello, dataset, dataset_pulito, encoder, low_final, span_final

# --- CARICAMENTO DATI ---
try:
    modello, dataset_raw, dataset_origin, encoder, low_final, span_final = carica_modello_e_dati()
    squadre_disponibili = sorted(dataset_origin['home_team'].unique())
except Exception as e:
    st.error(f"Errore nel caricamento dei dati. Assicurati che i file esistano. Dettaglio: {e}")
    st.stop()

# --- INTERFACCIA UTENTE ---
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Squadra in Casa", squadre_disponibili, index=squadre_disponibili.index('Juventus') if 'Juventus' in squadre_disponibili else 0)
with col2:
    away_team = st.selectbox("Squadra in Trasferta", squadre_disponibili, index=squadre_disponibili.index('Napoli') if 'Napoli' in squadre_disponibili else 1)

if st.button("Calcola Pronostico", type="primary", use_container_width=True):
    if home_team == away_team:
        st.warning("Seleziona due squadre diverse!")
    else:
        with st.spinner("Calcolo in corso..."):
            home_id = encoder.transform([home_team])[0]
            away_id = encoder.transform([away_team])[0]

            cols_to_drop = ['game', 'date']
            if 'target' in dataset_origin.columns:
                cols_to_drop.append('target')
            feature_columns = dataset_origin.drop(columns=cols_to_drop).columns
            
            home_history = dataset_origin[dataset_origin['home_team'] == home_team].iloc[-1]
            away_history = dataset_origin[dataset_origin['away_team'] == away_team].iloc[-1]
            
            matchup = dataset_raw[(dataset_raw['home_team'] == home_team) & (dataset_raw['away_team'] == away_team)]
            valore_h2h_home = matchup['home_goals'].tail(5).mean() if len(matchup) > 0 else 0.0
            valore_h2h_away = matchup['away_goals'].tail(5).mean() if len(matchup) > 0 else 0.0

            match_features = {
                'home_team': home_id, 'away_team': away_id,
                'h2h_home_goals_avg': valore_h2h_home, 'h2h_away_goals_avg': valore_h2h_away
            }
            
            for col in feature_columns:
                if 'home_avg' in col and 'gap' not in col: match_features[col] = home_history[col]
                elif 'away_avg' in col and 'gap' not in col: match_features[col] = away_history[col]

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
                probabilita = F.softmax(modello(input_tensor), dim=1).cpu().numpy()[0]
            
            p_1, p_x, p_2 = probabilita[0]*100, probabilita[1]*100, probabilita[2]*100
            
            st.markdown("### Probabilità")
            metric_cols = st.columns(3)
            metric_cols[0].metric(label="1 (Vittoria Casa)", value=f"{p_1:.1f}%")
            metric_cols[1].metric(label="X (Pareggio)", value=f"{p_x:.1f}%")
            metric_cols[2].metric(label="2 (Vittoria Trasferta)", value=f"{p_2:.1f}%")
            
            segni = ['1', 'X', '2']
            pronostico_finale = segni[np.argmax(probabilita)]
            st.success(f"**Pronostico Consigliato: {pronostico_finale}**")
