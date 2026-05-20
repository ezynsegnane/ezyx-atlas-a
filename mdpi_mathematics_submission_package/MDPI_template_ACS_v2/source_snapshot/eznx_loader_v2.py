# eznx_loader_v2.py - VERSION CORRIGÉE
import ast
from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb

DS5_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

META_FEATURES = [
    "age_z", "sex01", "height_z", "weight_z", "bmi_z",
    "miss__height", "miss__weight", "miss__bmi"
]
MASK_FEATURES = [
    "mask__age", "mask__sex", "mask__height", "mask__weight", "mask__bmi",
    "mask__miss_height", "mask__miss_weight", "mask__miss_bmi"
]

def _load_label_mapping(data_root: Path) -> Dict[str, str]:
    scp_path = data_root / "scp_statements.csv"
    if not scp_path.exists(): 
        scp_path = data_root.parent / "scp_statements.csv"
    if not scp_path.exists(): 
        return {}
    
    df_map = pd.read_csv(scp_path)
    mapping = {}
    for _, row in df_map.iterrows():
        if row['diagnostic_class'] in DS5_LABELS:
            mapping[str(row.iloc[0])] = row['diagnostic_class']
    return mapping

def _row_to_ds5_multi_hot(scp_codes_str: str, mapping: Dict[str, str]) -> np.ndarray:
    y = np.zeros(len(DS5_LABELS), dtype=np.float32)
    try:
        codes = ast.literal_eval(scp_codes_str)
        if isinstance(codes, dict): 
            codes = list(codes.keys())
    except:
        return y
        
    for code in codes:
        if str(code) in mapping:
            cls = mapping[str(code)]
            idx = DS5_LABELS.index(cls)
            y[idx] = 1.0
    return y

class EZNXDataset(Dataset):
    def __init__(self, index_file, data_root, fold=None, sampling_rate=100, meta_mode="demo+anthro"):
        self.data_root = Path(data_root)
        self.sampling_rate = sampling_rate
        self.meta_mode = meta_mode
        
        # 1. Pilotage automatique de la résolution
        if sampling_rate == 100:
            self.T = 1000
            col_name = 'filename_lr'
        else:
            self.T = 5000
            col_name = 'filename_hr'
        
        # 2. Chargement flexible de l'index (FIX pour Stage 2)
        if isinstance(index_file, pd.DataFrame):
            df = index_file.copy()
        else:
            df = pd.read_parquet(index_file)
        
        # 3. Filtrage par fold si demandé
        if fold is not None:
            df = df[df['strat_fold'] == fold].reset_index(drop=True)
            
        self.df = df
        
        # 4. Préparation des chemins
        self.recs = []
        for _, row in df.iterrows():
            fn = row[col_name]
            if fn.endswith('.hea'): 
                fn = fn[:-4]
            self.recs.append(str(self.data_root / fn))

        # 5. Labels et Métadonnées
        self.mapping = _load_label_mapping(self.data_root)
        self.y = np.array([_row_to_ds5_multi_hot(x, self.mapping) for x in df['scp_codes']])
        
        # ✅ CHARGEMENT DES DONNÉES BRUTES (avant ablation)
        self.meta = df[META_FEATURES].astype(np.float32).values.copy()
        self.mask = df[MASK_FEATURES].astype(np.float32).values.copy()

        self.meta = np.nan_to_num(self.meta, nan=0.0, posinf=0.0, neginf=0.0)
        self.mask = np.nan_to_num(self.mask, nan=0.0, posinf=0.0, neginf=0.0)
        self.mask = np.clip(self.mask, 0.0, 1.0)
        
        # 6. 🔧 GESTION DES MODES D'ABLATION (VERSION CORRIGÉE)
        if meta_mode == "none":
            # Mode NONE : Neutraliser tout
            self.meta = np.zeros_like(self.meta)
            self.mask = np.zeros_like(self.mask)
            
        elif meta_mode == "demo":
            # Mode DEMO : Garder seulement Age + Sex
            
            # ✅ ÉTAPE 1 : Neutraliser les VALEURS anthropométriques (Height, Weight, BMI)
            # Indices 2, 3, 4 → mettre à 0.0 (valeur standardisée "moyenne")
            self.meta[:, 2] = 0.0  # height_z
            self.meta[:, 3] = 0.0  # weight_z
            self.meta[:, 4] = 0.0  # bmi_z
            
            # ✅ ÉTAPE 2 : Forcer les FLAGS 'missing' à 1.0 (signaler l'ablation)
            # Indices 5, 6, 7 → mettre à 1.0 pour indiquer "donnée manquante"
            self.meta[:, 5] = 1.0  # miss__height
            self.meta[:, 6] = 1.0  # miss__weight
            self.meta[:, 7] = 1.0  # miss__bmi
            
            # ✅ ÉTAPE 3 : Désactiver le MASQUE d'attention pour les colonnes ablées
            # Les bits age/sex conservent leur disponibilité réelle.
            self.mask[:, 2] = 0.0  # mask__height (désactivé)
            self.mask[:, 3] = 0.0  # mask__weight (désactivé)
            self.mask[:, 4] = 0.0  # mask__bmi (désactivé)
            self.mask[:, 5] = 0.0  # mask__miss_height (désactivé)
            self.mask[:, 6] = 0.0  # mask__miss_weight (désactivé)
            self.mask[:, 7] = 0.0  # mask__miss_bmi (désactivé)
            
        # Mode "demo+anthro" : pas de modification, on garde tout
        # (les données restent telles quelles)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        try:
            sig, _ = wfdb.rdsamp(self.recs[i])
        except Exception as exc:
            raise RuntimeError(f"Failed to load WFDB record '{self.recs[i]}'") from exc

        L = sig.shape[0]
        if L < self.T:
            pad = np.zeros((self.T - L, 12), dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)
        elif L > self.T:
            sig = sig[:self.T, :]
            
        sig = sig.transpose()  # (12, T)
        # NOTE: The /5.0 voltage normalisation is applied in the training collate
        # functions (collate_fn_augmented and collate_fn_val in atlas_a_v5_multiseed.py
        # and atlas_a_v5_extended.py), NOT here.  This loader returns raw wfdb values
        # (in mV).  Do NOT apply /5.0 here to avoid double-scaling when used with
        # the full training pipeline.  See source_snapshot/README.md for details.

        return {
            "x_ts": torch.tensor(sig, dtype=torch.float32),
            "x_meta": torch.tensor(self.meta[i], dtype=torch.float32),
            "meta_present_mask": torch.tensor(self.mask[i], dtype=torch.float32),
            "y": torch.tensor(self.y[i], dtype=torch.float32),
            "ecg_id": self.df.iloc[i]['ecg_id']
        }
