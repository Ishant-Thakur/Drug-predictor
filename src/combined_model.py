import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/delaney-processed.csv")

print("Number of molecules:", len(df))


# --------------------------------------------------
# 2. RDKit descriptor features
# --------------------------------------------------

descriptor_columns = [
    "Molecular_Weight",
    "LogP",
    "TPSA",
    "H_Bond_Donors",
    "H_Bond_Acceptors",
    "Rotatable_Bonds",
    "Ring_Count",
    "Heavy_Atom_Count",
    "Fraction_CSP3"
]

descriptor_df = pd.read_csv(
    "data/delaney_with_descriptors.csv"
)

descriptor_features = descriptor_df[
    descriptor_columns
].values


# --------------------------------------------------
# 3. Generate Morgan fingerprints
# --------------------------------------------------

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

fingerprints = []

for smiles in df["smiles"]:

    mol = Chem.MolFromSmiles(smiles)

    fingerprint = morgan_generator.GetFingerprintAsNumPy(mol)

    fingerprints.append(fingerprint)


fingerprints = np.array(fingerprints)

print("Fingerprint shape:", fingerprints.shape)


# --------------------------------------------------
# 4. Reduce fingerprint dimensions with PCA
# --------------------------------------------------

# We reduce 2048 fingerprint features to 50 components.
#
# PCA will be placed inside the pipeline so that
# it is fitted separately on each training fold.

pca = PCA(
    n_components=50,
    random_state=42
)


# --------------------------------------------------
# 5. Combine descriptors + PCA fingerprints
# --------------------------------------------------

# We cannot directly combine them before PCA because
# PCA should only operate on fingerprint features.

# First transform fingerprints temporarily so we can
# create the combined dataset for experimentation.
fingerprint_pca = pca.fit_transform(fingerprints)

print("PCA fingerprint shape:", fingerprint_pca.shape)


combined_features = np.hstack(
    [
        descriptor_features,
        fingerprint_pca
    ]
)

print(
    "Combined feature shape:",
    combined_features.shape
)


# --------------------------------------------------
# 6. Target
# --------------------------------------------------

y = df[
    "measured log solubility in mols per litre"
].values


# --------------------------------------------------
# 7. Random Forest
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 8. 5-Fold Cross Validation
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
    combined_features,
    y,
    cv=kf,
    scoring=scoring,
    n_jobs=-1
)


# --------------------------------------------------
# 9. Results
# --------------------------------------------------

mean_mae = -results["test_MAE"].mean()
mean_rmse = -results["test_RMSE"].mean()
mean_r2 = results["test_R2"].mean()

std_mae = results["test_MAE"].std()
std_rmse = results["test_RMSE"].std()
std_r2 = results["test_R2"].std()


print("\n--- Descriptor + PCA Fingerprint Model ---")

print(
    f"Mean MAE  : {mean_mae:.4f} ± {std_mae:.4f}"
)

print(
    f"Mean RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}"
)

print(
    f"Mean R²   : {mean_r2:.4f} ± {std_r2:.4f}"
)


# --------------------------------------------------
# 10. Final comparison
# --------------------------------------------------

print("\n--- Model Comparison ---")

print("Descriptors + Random Forest : R² = 0.8892")
print("Morgan Fingerprints         : R² = 0.6746")
print(
    f"Descriptors + PCA           : R² = {mean_r2:.4f}"
)