import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

def _default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", required=False, default=20, type=int)
args = parser.parse_args()

def prepare_gamma_data(df_gamma, mode='split', val_size=0.2, device=None, verbose=False):
    if device is None:
        device = _default_device()

    if mode == 'split':
        train_ids, val_ids = train_test_split(df_gamma['system_ID'].unique(), test_size=val_size, random_state=args.seed)
        datasets = {
            'train': df_gamma[df_gamma['system_ID'].isin(train_ids)],
            'val': df_gamma[df_gamma['system_ID'].isin(val_ids)]
        }

        results = {}
        for name, df_subset in datasets.items():
            X = torch.tensor(df_subset[['T', 'x1']].values, dtype=torch.float32, device=device)
            bert1 = torch.stack(list(df_subset['BERT_component_1_ID']), dim=0).to(dtype=torch.float32, device=device)
            bert2 = torch.stack(list(df_subset['BERT_component_2_ID']), dim=0).to(dtype=torch.float32, device=device)
            X = torch.cat([X, bert1, bert2], dim=1)

            y = torch.tensor(df_subset[['ln_gamma_1', 'ln_gamma_2']].values, dtype=torch.float32, device=device)
            systems_ID = torch.tensor(df_subset['system_ID'].values, dtype=torch.int64, device=device)

            results[f'X_{name}'] = X
            results[f'y_{name}'] = y
            results[f'{name}_systems_ID'] = systems_ID

            if verbose:
                print(f"{name} set: {df_subset['component_1_ID'].nunique()} Komponenten, {df_subset['system_ID'].nunique()} Systene, {df_subset.shape[0]} Datenpunkte")

        return results

    elif mode == 'full':
        X = torch.tensor(df_gamma[['T', 'x1']].values, dtype=torch.float32, device=device)
        bert1 = torch.stack(list(df_gamma['BERT_component_1_ID']), dim=0).to(dtype=torch.float32, device=device)
        bert2 = torch.stack(list(df_gamma['BERT_component_2_ID']), dim=0).to(dtype=torch.float32, device=device)
        X = torch.cat([X, bert1, bert2], dim=1)

        y = torch.tensor(df_gamma[['ln_gamma_1', 'ln_gamma_2']].values, dtype=torch.float32, device=device)
        systems_ID = df_gamma['system_ID'].values.astype(np.int64)
        component_1_ID = df_gamma['component_1_ID'].values.astype(np.int64)
        component_2_ID = df_gamma['component_2_ID'].values.astype(np.int64)
        y1 = df_gamma['y1'].values.astype(np.float32)
        ps1 = df_gamma['ps1'].values.astype(np.float32)
        ps2 = df_gamma['pS2'].values.astype(np.float32)

        with open("data/component_ID_to_SMILES.pkl", "rb") as f:
            int_to_smiles = pickle.load(f)

        if verbose:
            print(f"full set: {df_gamma['component_1_ID'].nunique()} Komponenten, {df_gamma['system_ID'].nunique()} Systene, {df_gamma.shape[0]} Datenpunkte")

        return {
            'X': X,
            'y': y,
            'systems_ID': systems_ID,
            'component_1_ID': component_1_ID,
            'component_2_ID': component_2_ID,
            'int_to_smiles': int_to_smiles,
            'y1': y1,
            'ps1': ps1,
            'ps2': ps2,
        }

    else:
        raise ValueError("mode muss 'split' oder 'full' sein.")

#---------------------------------------------------------------------------------------------------------------------------------------------------------

def preprocess_input(x, Embedding_BERT, device=None):
    if device is None:
        device = _default_device()
    
    num_samples = x.shape[0]
    num_components = (x.shape[1] - 1) // Embedding_BERT

    #Shape-Check
    expected_columns = 1 + (num_components - 1) + num_components * Embedding_BERT
    assert x.shape[1] == expected_columns, f"Input shape doesn't match the expected shape based on Embedding_BERT. Expected {expected_columns} columns, got {x.shape[1]}"

    #Extrahieren der Temperatur
    T_x = x[:, :1]  # Shape: [num_samples, 1]

    #Erstellen eines leeren Tensor für die umgeformten Daten
    reshaped_data = torch.empty((num_samples, num_components, Embedding_BERT + 2), device=device)

    #Temperatur für alle Komponenten füllen
    reshaped_data[:, :, 0] = T_x

    #Molekülfraktion berechnen
    for i in range(num_components):
        if i != num_components - 1:
            reshaped_data[:, i, 1] = x[:, i+1]
        else:
            reshaped_data[:, i, 1] = 1 - torch.sum(reshaped_data[:, :-1, 1], dim=1)

        #Embeddings extrahieren und speichern
        start_idx = 1 + num_components - 1 + i * Embedding_BERT
        end_idx = start_idx + Embedding_BERT
        reshaped_data[:, i, 2:] = x[:, start_idx:end_idx]

    return reshaped_data

#---------------------------------------------------------------------------------------------------------------------------------------------------------

def split_and_reshape_input(input_array):

    #Extrahieren der Temperatur
    global_temperature = input_array[:, 0, 0]  #Shape: [batch_size]

    #Extrahieren der Molekülfraktionen
    mole_fractions_N_minus_1 = input_array[:, 0, 1:2]  #Shape: [batch_size, 1]

    #Extrahieren der Embeddings
    feature_points = input_array[:, :, 2:].contiguous()  #Shape: [batch_size, num_components, Embedding_BERT]

    return global_temperature, mole_fractions_N_minus_1, feature_points
