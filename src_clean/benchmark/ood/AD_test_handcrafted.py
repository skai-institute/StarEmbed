import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_from_disk
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


input_embs=(
"/projects/b1094/StarEmbed/embeddings/csdr1_raw4_catflags_filtered_embs_hand_crafted_trn_val_tst_bandgr",
)


seeds = [0, 42, 123]
param_grid = {
        'n_estimators': [100, 200],
}

def add_embedding(example):

    emb_g = np.squeeze(np.array(example["embeddings_g"], dtype=np.float32))
    emb_r = np.squeeze(np.array(example["embeddings_r"], dtype=np.float32))

    if emb_g.ndim > 1:
        avg_g, avg_r = emb_g.mean(0), emb_r.mean(0)
    else:
        avg_g, avg_r = emb_g, emb_r

    example['g_embedding'] = avg_g
    example['r_embedding'] = avg_r

    return example

def add_embedding_batch(examples):
    arr_g = np.array(examples["embeddings_g"], dtype=np.float32)  # shape: (B, …, D)
    arr_r = np.array(examples["embeddings_r"], dtype=np.float32)

    # drop any spurious 1–length dimensions before collapsing
    # e.g. (B, 1, T, D) → (B, T, D)
    if arr_g.ndim == 4 and arr_g.shape[1] == 1:
        arr_g = arr_g.squeeze(axis=1)
        arr_r = arr_r.squeeze(axis=1)

    # now arr_g.ndim is either 2 (B, D) or 3 (B, T, D)
    if arr_g.ndim == 3:
        g = arr_g.mean(axis=1)
        r = arr_r.mean(axis=1)
    else:
        g, r = arr_g, arr_r

    return {"g_embedding": g, "r_embedding": r}

def main(state):

    for f in input_embs:

        print(f)
        
        ds = load_from_disk(f, keep_in_memory=False)
        ds_anom = load_from_disk("/projects/b1094/rehemtulla/SkAI/skai_universal_forecaster/data/embs/csdr1_raw4_catflags_filtered_embs_anom_bandgr")

        #print(dataset['train'].features)

        # Convenient arrow -> NumPy view (avoids a full copy)
        if 'embeddings_g' in ds['validation'].features:
            ds = ds.map(
                            add_embedding_batch,
                            batched=True,
                            batch_size=512,                     
                            num_proc=4,                         # multiprocessing gives diminishing returns here
                            remove_columns=["embeddings_g", "embeddings_r"],
                            keep_in_memory=False,               # let Hugging Face handle caching on disk
                        )

        ds.set_format(type="numpy", columns=["g_embedding","r_embedding","class_str"])
        ds_anom.set_format(type="numpy", columns=["g_embedding","r_embedding","class_str"])

        def batched_xy(split):
            """
            Returns X, y as 2-D (n_samples, dim) and 1-D (n_samples,) NumPy arrays.
            Hugging Face datasets guarantee that ds[split]['embedding'] already
            comes back as an (n_samples, dim) array after set_format().
            """
            if split == 'anoms':
                X = np.concatenate([ds_anom[split]["g_embedding"], ds_anom[split]["r_embedding"]], 1)# (n_samples, embed_dim)
                y = ds_anom[split]["class_str"]             
            else:
                X = np.concatenate([ds[split]["g_embedding"], ds[split]["r_embedding"]], 1)# (n_samples, embed_dim)
                y = ds[split]["class_str"]                      # already 1-D
            return X, y

        X_train, y_train = batched_xy("train")
        X_val,   y_val   = batched_xy("validation")
        X_test,  y_test  = batched_xy("test")
        X_anom,  y_anom  = batched_xy("anoms")

        print(X_test.shape, X_anom.shape)

        X_combined_test = np.concatenate([X_test, X_anom])
        y_combined_test = np.concatenate([y_test, y_anom])

        print(y_combined_test.shape)

        models = {}

        # Fit the isolation forest to all the majority classes
        for i in np.unique(np.asarray(y_val)):
            idx = np.where(np.asarray(y_val)==i)[0]
            m = IsolationForest(n_estimators=200, random_state=state)
            m.fit(X_train[idx, :])
            models[i] = m

        # Compute the anomaly scores
        df = pd.DataFrame()
        df['True'] = y_combined_test
        for key in models:
            df[key] = models[key].decision_function(X_combined_test)

        df.to_csv(f"OOD/{f.split('/')[-1]}_AD_{seed}.csv", index=False)       

        




if __name__=='__main__':
    for seed in seeds:
        main(seed)



