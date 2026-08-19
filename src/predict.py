import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "attention_mlp_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "attention_mlp_scaler.pkl")


# --------------------------------------------------
# 2. Feature list + architecture
#    (must match 01_attention_mlp.ipynb exactly, or the
#    saved state_dict will fail to load)
# --------------------------------------------------

FEATURES = [
    "Molecular_Weight", "LogP", "TPSA", "H_Bond_Donors", "H_Bond_Acceptors",
    "Rotatable_Bonds", "Ring_Count", "Heavy_Atom_Count", "Fraction_CSP3",
    "Aromatic_Rings", "Molar_Refractivity", "Num_Heteroatoms",
    "Num_Aromatic_Carbocycles", "BalabanJ", "Labute_ASA",
]

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 2
MLP_HIDDEN = 64
DROPOUT = 0.1

scaler = joblib.load(SCALER_PATH)


class AttentionMLP(nn.Module):
    """Feature self-attention model. Each of the 12 descriptors is
    embedded into its own token (FT-Transformer style), a learnable
    [CLS] token is prepended, and a Transformer encoder lets
    descriptors attend to each other before pooling into the shared
    trunk and three output heads."""

    def __init__(self, n_features, d_model=32, n_heads=4, n_layers=2, mlp_hidden=64, dropout=0.1):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.ReLU(),
            nn.LayerNorm(mlp_hidden), nn.Dropout(dropout),
        )
        self.solubility_head = nn.Linear(mlp_hidden, 1)
        self.bbb_head = nn.Linear(mlp_hidden, 1)
        self.tox_head = nn.Linear(mlp_hidden, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(seq)
        h = self.trunk(encoded[:, 0, :])

        return (
            self.solubility_head(h).squeeze(-1),
            self.bbb_head(h).squeeze(-1),
            self.tox_head(h).squeeze(-1),
        )


model = AttentionMLP(
    n_features=len(FEATURES),
    d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
    mlp_hidden=MLP_HIDDEN, dropout=DROPOUT,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# --------------------------------------------------
# 3. Descriptor calculation (must match training exactly)
# --------------------------------------------------

def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    descriptors = {
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "H_Bond_Donors": Lipinski.NumHDonors(mol),
        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring_Count": Lipinski.RingCount(mol),
        "Heavy_Atom_Count": Lipinski.HeavyAtomCount(mol),
        "Fraction_CSP3": Lipinski.FractionCSP3(mol),
        "Aromatic_Rings": Lipinski.NumAromaticRings(mol),
        "Molar_Refractivity": Crippen.MolMR(mol),
        "Num_Heteroatoms": Lipinski.NumHeteroatoms(mol),
        "Num_Aromatic_Carbocycles": Lipinski.NumAromaticCarbocycles(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "Labute_ASA": Descriptors.LabuteASA(mol),
    }

    return descriptors


# --------------------------------------------------
# 4. Predict solubility + BBB + toxicity
#    (same public function name/shape as before, so app.py
#    needs no changes)
# --------------------------------------------------

def predict_properties(smiles):

    descriptors = calculate_descriptors(smiles)

    x = np.array([[descriptors[f] for f in FEATURES]])
    x = scaler.transform(x)
    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        p_logs, p_bbb, p_tox = model(x)

    bbb_prob = float(torch.sigmoid(p_bbb).item())
    tox_prob = float(torch.sigmoid(p_tox).item())

    return {
        "predicted_logS": float(p_logs.item()),
        "bbb_probability": bbb_prob,
        "bbb_label": "Permeant" if bbb_prob >= 0.5 else "Non-permeant",
        "toxicity_probability": tox_prob,
        "toxicity_label": "Likely toxic" if tox_prob >= 0.5 else "Likely non-toxic",
        "descriptors": descriptors,
    }


def predict_solubility(smiles):
    """Kept for backward compatibility: returns (logS, descriptors)."""
    result = predict_properties(smiles)
    return result["predicted_logS"], result["descriptors"]


# --------------------------------------------------
# 5. Test the prediction system
# --------------------------------------------------

if __name__ == "__main__":

    for smiles in ["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]:

        result = predict_properties(smiles)

        print(f"\n--- {smiles} ---")
        print(f"Predicted LogS   : {result['predicted_logS']:.4f}")
        print(f"BBB permeability : {result['bbb_label']} (p={result['bbb_probability']:.3f})")
        print(f"Toxicity risk    : {result['toxicity_label']} (p={result['toxicity_probability']:.3f})")