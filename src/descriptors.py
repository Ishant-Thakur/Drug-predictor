import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen


# ---------------------------------------------
# 1. Load dataset
# ---------------------------------------------

df = pd.read_csv("data/delaney-processed.csv")


# ---------------------------------------------
# 2. Function to calculate descriptors
# ---------------------------------------------

def calculate_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    # Invalid SMILES
    if mol is None:
        return None

    return {
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "H_Bond_Donors": Lipinski.NumHDonors(mol),
        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring_Count": Lipinski.RingCount(mol),
        "Heavy_Atom_Count": Lipinski.HeavyAtomCount(mol),
        "Fraction_CSP3": Lipinski.FractionCSP3(mol)
    }


# ---------------------------------------------
# 3. Generate descriptors
# ---------------------------------------------

descriptor_data = []

invalid_smiles = 0

for smiles in df["smiles"]:

    descriptors = calculate_descriptors(smiles)

    if descriptors is None:
        invalid_smiles += 1
        descriptor_data.append({})
    else:
        descriptor_data.append(descriptors)


# ---------------------------------------------
# 4. Convert descriptors to DataFrame
# ---------------------------------------------

descriptor_df = pd.DataFrame(descriptor_data)


# ---------------------------------------------
# 5. Combine with original dataset
# ---------------------------------------------

df_final = pd.concat(
    [df, descriptor_df],
    axis=1
)


# ---------------------------------------------
# 6. Print results
# ---------------------------------------------

print("\n--- Descriptor Dataset ---")
print(df_final.head())

print("\n--- Dataset Shape ---")
print(df_final.shape)

print("\n--- Descriptor Columns ---")
print(descriptor_df.columns.tolist())

print("\n--- Invalid SMILES ---")
print(invalid_smiles)


# ---------------------------------------------
# 7. Save new dataset
# ---------------------------------------------

df_final.to_csv(
    "data/delaney_with_descriptors.csv",
    index=False
)

print("\nSaved to:")
print("data/delaney_with_descriptors.csv")