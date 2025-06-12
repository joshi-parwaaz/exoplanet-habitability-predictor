import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data")
    raw_path = os.path.join(data_dir, "exoplanets_processed.csv")
    artifact_dir = os.path.join(data_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # 1) Load processed data
    df = pd.read_csv(raw_path)
    X = df.drop("habitable", axis=1).values
    y = df["habitable"].values

    # 2) Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 3) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # 4) Save artifacts
    #   - NumPy arrays for model input
    #   - Scaler for later inference
    np.save(os.path.join(artifact_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(artifact_dir, "X_val.npy"),   X_val_scaled)
    np.save(os.path.join(artifact_dir, "y_train.npy"), y_train)
    np.save(os.path.join(artifact_dir, "y_val.npy"),   y_val)
    joblib.dump(scaler, os.path.join(artifact_dir, "scaler.joblib"))

    print("Preprocessing complete.")
    print(f"  Train set: X_train {X_train_scaled.shape}, y_train {y_train.shape}")
    print(f"  Val   set: X_val   {X_val_scaled.shape}, y_val   {y_val.shape}")
    print(f"Artifacts saved to {artifact_dir}")

if __name__ == "__main__":
    main()
