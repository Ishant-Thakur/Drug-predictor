import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/delaney-processed.csv")

print("Number of molecules:", len(df))


# --------------------------------------------------
# 2. Create Morgan fingerprint generator
# --------------------------------------------------

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


# --------------------------------------------------
# 3. Convert SMILES → Morgan fingerprints
# --------------------------------------------------

fingerprints = []

invalid_smiles = 0

for smiles in df["smiles"]:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        invalid_smiles += 1
        fingerprints.append(np.zeros(2048, dtype=np.uint8))
        continue

    fingerprint = morgan_generator.GetFingerprintAsNumPy(mol)

    fingerprints.append(fingerprint)


# --------------------------------------------------
# 4. Convert to NumPy array
# --------------------------------------------------

X = np.array(fingerprints)

y = df[
    "measured log solubility in mols per litre"
].values


print("\nFingerprint shape:", X.shape)
print("Invalid SMILES:", invalid_smiles)


# --------------------------------------------------
# 5. Random Forest
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 6. 5-Fold Cross Validation
# --------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}


results = cross_validate(
    model,
    X,
    y,
    cv=kf,
    scoring=scoring,
    n_jobs=-1
)


# --------------------------------------------------
# 7. Results
# --------------------------------------------------

mean_mae = -results["test_MAE"].mean()
mean_rmse = -results["test_RMSE"].mean()
mean_r2 = results["test_R2"].mean()

std_mae = results["test_MAE"].std()
std_rmse = results["test_RMSE"].std()
std_r2 = results["test_R2"].std()


print("\n--- Morgan Fingerprint Random Forest ---")

print(
    f"Mean MAE  : {mean_mae:.4f} ± {std_mae:.4f}"
)

print(
    f"Mean RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}"
)

print(
    f"Mean R²   : {mean_r2:.4f} ± {std_r2:.4f}"
)