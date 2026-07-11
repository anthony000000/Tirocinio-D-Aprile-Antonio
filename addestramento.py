import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "..", "nuovo_event_log")
PATH_PICKLE = os.path.join(INPUT_FOLDER, "prepared_dataset.pkl")

TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Iperparametri XGBoost centralizzati
XGB_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',
    'n_estimators': 100,
    'random_state': RANDOM_SEED,
    'eval_metric': 'mlogloss'
}

# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Carica il pickle, ordina e rimuove allergie."""
    print("1/4 - Caricamento e pulizia dati...")
    df = pd.read_pickle(filepath)
    df = df.sort_values(by=['PATIENT', 'Elapsed_Time_Days']).reset_index(drop=True)
    
    cols_to_drop = [col for col in df.columns if 'allergy' in col.lower()]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def temporal_train_test_split(df: pd.DataFrame, ratio: float):
    """Divide i dati per paziente rispettando la cronologia."""
    print(f"2/4 - Esecuzione Split Temporale ({int(ratio*100)}/{int((1-ratio)*100)}) per Paziente...")
    train_list, test_list = [], []

    for _, group in df.groupby('PATIENT'):
        split_idx = int(len(group) * ratio)
        if split_idx == 0:
            train_list.append(group)
        else:
            train_list.append(group.iloc[:split_idx])
            test_list.append(group.iloc[split_idx:])

    return pd.concat(train_list), pd.concat(test_list)

def encode_and_filter_labels(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Mantiene solo le classi condivise e applica il Label Encoding."""
    train_classes = set(df_train['Target_Next_Activity'].unique())
    test_classes = set(df_test['Target_Next_Activity'].unique())
    valid_classes = list(train_classes.intersection(test_classes))

    df_train = df_train[df_train['Target_Next_Activity'].isin(valid_classes)]
    df_test = df_test[df_test['Target_Next_Activity'].isin(valid_classes)]

    le = LabelEncoder()
    le.fit(valid_classes)
    
    y_train = le.transform(df_train['Target_Next_Activity'])
    y_test = le.transform(df_test['Target_Next_Activity'])
    
    return df_train, df_test, y_train, y_test, le

# ==========================================
# FLUSSO PRINCIPALE (MAIN)
# ==========================================
def main():
    # 1. Caricamento e Pulizia
    df = load_and_clean_data(PATH_PICKLE)
    
    # 2. Split e Filtraggio
    df_train, df_test = temporal_train_test_split(df, TRAIN_SPLIT_RATIO)
    df_train, df_test, y_train, y_test, le = encode_and_filter_labels(df_train, df_test)

    # 3. Estrazione Matrici X
    X_train_full = df_train.drop(columns=['PATIENT', 'Target_Next_Activity'])
    X_test_full = df_test.drop(columns=['PATIENT', 'Target_Next_Activity'])

    # Calcolo pesi globali
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    target_names = le.inverse_transform(range(len(le.classes_)))

    # 4. Configurazioni
    context_cols = [col for col in X_train_full.columns if col.startswith('Context_')]
    time_cols = ['Elapsed_Time_Days']

    configurations = {
        "1. SOLO ACTIVITY": context_cols + time_cols,
        "2. ACTIVITY + TIME": context_cols,
        "3. ACTIVITY + TIME + CONTEXT": []
    }

    print("\n3/4 - Avvio Addestramento per le tre configurazioni")
    results = []

    for config_name, drop_cols in configurations.items():
        print(f"\n---> Addestramento: {config_name} <---")
        X_train_curr = X_train_full.drop(columns=drop_cols)
        X_test_curr = X_test_full.drop(columns=drop_cols)
        
        print(f"      Colonne attive: {X_train_curr.shape[1]}")
        
        # Inizializzazione modello con i parametri globali
        model = xgb.XGBClassifier(**XGB_PARAMS)
        
        # Training e Prediction
        model.fit(X_train_curr, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test_curr)
        
        # Metriche
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            "Configurazione": config_name,
            "Accuracy": f"{acc*100:.2f}%",
            "Macro F1": f"{macro_f1*100:.2f}%"
        })
        
        # Stampa report
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # 5. Risultato Finale
    print("\n4/4 - CONFRONTO FINALE:")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    main()