import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
from tqdm import tqdm  # Aggiunta per la barra di caricamento

# ==========================================
# CONFIGURAZIONI GLOBALI
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "..", "nuovo_event_log")
PATH_INPUT = os.path.join(INPUT_FOLDER, "synthea_event_log.csv")
PATH_OUTPUT = os.path.join(INPUT_FOLDER, "prepared_dataset.pkl")

VECTOR_SIZE = 128
WINDOW_SIZE = 5
MIN_COUNT = 1
WORKERS = 4

# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================
def load_and_sort_data(filepath: str) -> pd.DataFrame:
    """Carica il dataset e lo ordina cronologicamente per paziente."""
    print("1/4 - Caricamento del dataset grezzo in memoria...")
    df = pd.read_csv(filepath, low_memory=False)
    df['start_date'] = pd.to_datetime(df['start_date'])
    return df.sort_values(by=['PATIENT', 'start_date']).reset_index(drop=True)

def train_word2vec(df: pd.DataFrame) -> Word2Vec:
    """Addestra il modello Word2Vec sui contesti medici."""
    print(f"2/4 - Addestramento Word2Vec sui Context (Dimensione: {VECTOR_SIZE})...")
    sentences = df.groupby('PATIENT')['DESCRIPTION_enc'].apply(
        lambda x: x.fillna('Unknown').astype(str).tolist()
    ).tolist()
    
    model = Word2Vec(
        sentences=sentences, 
        vector_size=VECTOR_SIZE, 
        window=WINDOW_SIZE, 
        min_count=MIN_COUNT, 
        workers=WORKERS
    )
    return model

def get_average_vector(words: list, model: Word2Vec, size: int) -> np.ndarray:
    """Calcola la media matematica dei vettori Word2Vec di una lista di parole."""
    valid_words = [word for word in words if word in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(size)
    vectors = [model.wv[word] for word in valid_words]
    return np.mean(vectors, axis=0)

def build_features(df: pd.DataFrame, w2v_model: Word2Vec):
    """Estrae le features (Activity Count, Word2Vec Context, Time) iterando sui pazienti."""
    print("3/4 - Estrazione features e creazione finestre temporali...")
    
    patient_ids, elapsed_times, next_activities = [], [], []
    activity_counts_list, context_vectors_list = [], []

    # tqdm crea una barra di caricamento nel terminale
    total_patients = df['PATIENT'].nunique()
    for patient_id, group in tqdm(df.groupby('PATIENT'), total=total_patients, desc="Elaborazione Pazienti"):
        activities = group['encounter_type'].tolist()
        dates = group['start_date'].tolist()
        contexts = group['DESCRIPTION_enc'].fillna('Unknown').astype(str).tolist()
        
        if len(activities) >= 2:
            for i in range(1, len(activities)):
                # 1. Frequency Encoding
                activity_counts_list.append(dict(Counter(activities[:i])))
                
                # 2. Vettorializzazione Word2Vec
                context_vectors_list.append(get_average_vector(contexts[:i], w2v_model, VECTOR_SIZE))
                
               # 3. Tempo Trascorso (Nuova versione in ORE)
                elapsed_times.append((dates[i-1] - dates[0]).total_seconds() / 3600.0)
                
                # 4. Target & ID
                next_activities.append(activities[i])
                patient_ids.append(patient_id)
                
    return patient_ids, elapsed_times, next_activities, activity_counts_list, context_vectors_list

# ==========================================
# FLUSSO PRINCIPALE (MAIN)
# ==========================================
def main():
    # 1. Caricamento
    df = load_and_sort_data(PATH_INPUT)
    
    # 2. Addestramento Word2Vec
    w2v_model = train_word2vec(df)
    
    # 3. Feature Engineering
    p_ids, times, targets, act_counts, ctx_vecs = build_features(df, w2v_model)
    
    # 4. Assemblaggio Dataset
    print("\n4/4 - Creazione del tabellone matematico finale...")
    df_base = pd.DataFrame({
        'PATIENT': p_ids,
        'Elapsed_Time_Hours': times,
        'Target_Next_Activity': targets
    })
    
    print("      -> Generazione colonne conteggi...")
    df_activities = pd.DataFrame(act_counts).fillna(0).astype(int)
    df_activities.columns = [f"Count_{col}" for col in df_activities.columns]
    
    print("      -> Generazione colonne Word2Vec...")
    df_context = pd.DataFrame(ctx_vecs, columns=[f'Context_V{j+1}' for j in range(VECTOR_SIZE)])
    
    df_prepared = pd.concat([df_base, df_activities, df_context], axis=1)
    df_prepared.to_pickle(PATH_OUTPUT)
    
    print("\n✅ OPERAZIONE COMPLETATA CON SUCCESSO")
    print(f"Dataset pronto e salvato in: {PATH_OUTPUT}")

# Questo blocco avvia lo script solo se viene eseguito direttamente
if __name__ == "__main__":
    main()