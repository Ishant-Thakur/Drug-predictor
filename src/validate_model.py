import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/delaney_with_descriptors.csv")


# --------------------------------------------------
# 2. Features and target
# --------------------------------------------------

features = [
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

target = "measured log solubility in mols per litre"

X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Create Random Forest
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 4. 5-Fold Cross Validation
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
    rf_model,
    X,
    y,
    cv=kf,
    scoring=scoring,
    return_train_score=True
)


# --------------------------------------------------
# 5. Print individual fold results
# --------------------------------------------------

print("\n--- 5-Fold Cross Validation ---")

for i in range(5):

    mae = -results["test_MAE"][i]
    rmse = -results["test_RMSE"][i]
    r2 = results["test_R2"][i]

    print(
        f"Fold {i + 1}: "
        f"MAE={mae:.4f}, "
        f"RMSE={rmse:.4f}, "
        f"R²={r2:.4f}"
    )


# --------------------------------------------------
# 6. Average CV results
# --------------------------------------------------

mean_mae = -results["test_MAE"].mean()
mean_rmse = -results["test_RMSE"].mean()
mean_r2 = results["test_R2"].mean()

std_mae = results["test_MAE"].std()
std_rmse = results["test_RMSE"].std()
std_r2 = results["test_R2"].std()


print("\n--- Cross Validation Summary ---")

print(f"Mean MAE  : {mean_mae:.4f} ± {std_mae:.4f}")
print(f"Mean RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
print(f"Mean R²   : {mean_r2:.4f} ± {std_r2:.4f}")


# --------------------------------------------------
# 7. Check for overfitting
# --------------------------------------------------

train_r2 = results["train_R2"].mean()
test_r2 = results["test_R2"].mean()

print("\n--- Overfitting Check ---")

print(f"Average Training R² : {train_r2:.4f}")
print(f"Average Testing R²  : {test_r2:.4f}")
print(f"R² Difference       : {train_r2 - test_r2:.4f}")


# --------------------------------------------------
# 8. ESOL baseline
# --------------------------------------------------

actual = df[target]

esol_predictions = df[
    "ESOL predicted log solubility in mols per litre"
]

esol_mae = mean_absolute_error(
    actual,
    esol_predictions
)

esol_rmse = mean_squared_error(
    actual,
    esol_predictions
) ** 0.5

esol_r2 = r2_score(
    actual,
    esol_predictions
)


print("\n--- ESOL Baseline ---")

print(f"MAE  : {esol_mae:.4f}")
print(f"RMSE : {esol_rmse:.4f}")
print(f"R²   : {esol_r2:.4f}")


# --------------------------------------------------
# 9. Final comparison
# --------------------------------------------------

print("\n--- Model Comparison ---")

print(
    f"Random Forest CV R² : {mean_r2:.4f}"
)

print(
    f"ESOL R²             : {esol_r2:.4f}"
)