import pandas as pd
import pickle
import sys; sys.path.append('../')
import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import os
    
def save_df(df: pd.DataFrame):
    df.to_csv(f"data/{df.name}.csv")
    
def check_cache(file):
    os.makedirs("data/cache", exist_ok=True)
    if f"{file}.csv" in [i for i in os.listdir("data/cache")]:
        return True
    else: 
        return False
    
def initiliaze_ChemBERTA(model_name="DeepChem/ChemBERTa-77M-MTR", device=None):
    # Load the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create the directory if it doesn't exist
    # os.makedirs('ChemBERTa', exist_ok=True)
    
    # Save the tokenizer's vocabulary to the specified folder
    # tokenizer.save_vocabulary('ChemBERTa/')
    
    # Define ChemBERTa model and move it to the specified device
    ChemBERTA = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name).to(device).eval()
    # ChemBERTA.save_pretrained('ChemBERTa')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return ChemBERTA, tokenizer

def batched_smiles_embedding(smiles, batch_size, tokenizer, model, device=None, max_length=512):
    embs = []
    for i in tqdm.tqdm( range(0, len(smiles), batch_size), desc="processing"):
        batch = smiles[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            cls = model(**enc)["last_hidden_state"][:, 0, :].cpu().numpy()
        embs.extend(cls[:, None, :])
    
    return embs

def get_embeddings(com_csv, tokenizer, model):
    
    if check_cache(com_csv):
        vle_df = pd.read(f"data/cache/vle_df.csv")
    else: 
        vle_df = pd.read_csv(com_csv)[["DDB", "Canonical_SMILES_RDkit"]]
        smiles_list = vle_df["Canonical_SMILES_RDkit"].to_list()
    
        vle_df['Embeddings'] = batched_smiles_embedding(smiles_list, batch_size=256, tokenizer=tokenizer, model=model, device="mps")
    return vle_df



if __name__ == "__main__": 
    model, tokenizer= initiliaze_ChemBERTA("DeepChem/ChemBERTa-100M-MLM", device="mps")

    embeddings = get_embeddings("data/original_data/Components_2025_SMILES_cleaned_with_ids.csv", tokenizer, model)
    save_df(embeddings)
