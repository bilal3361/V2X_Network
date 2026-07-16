import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf
import random
from datetime import datetime, timezone
from pathlib import Path

# =========================================================
# 0. Reproducibility
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =========================================================
# 1. Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "plots" / "training_outputs"
MODEL_DIR = PROJECT_ROOT / "models"

CSV_PATH = DATA_DIR / "vehicle_trajectory_dataset.csv"
OUTPUT_PRED_CSV = DATA_DIR / "trajectory_predictions.csv"

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_trajectory_model.keras")
FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler.joblib")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "task5_model_metadata.json")
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, "trajectory_training_validation_loss.png")
SINGLE_TRAJ_PLOT_PATH = os.path.join(OUTPUT_DIR, "trajectory_actual_vs_predicted_single_sample.png")
ALL_TRAJ_PLOT_PATH = os.path.join(OUTPUT_DIR, "all_vehicle_trajectories_overlay.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def project_relative_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()

# =========================================================
# 2. Time and model settings
# =========================================================
DT_SECONDS = 0.1

# 80 steps × 0.1 sec = 8 seconds of past trajectory
INPUT_LEN = 20

# 5 seconds future prediction
PRED_LEN = 30

# Generate one training sequence every 0.2 seconds.
# Keeps 0.1-second resolution inside each sequence.
SEQUENCE_STRIDE = 2

# Accuracy threshold for continuous point-wise accuracy
#ERROR_THRESHOLD_M = 5.0

# Toggle model training data region
# True  -> train/evaluate only on near-junction rows
# False -> train/evaluate on all rows in the dataset
USE_NEAR_JUNCTION_ONLY = True                                 #        Use True to consider the near vehichles which are in the radius 100m              

# =========================================================
# 3. Load dataset
# =========================================================
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# =========================================================
# 4. Basic cleaning
# =========================================================
print("\nMissing values BEFORE cleaning:")
print(df.isna().sum())

# Text / categorical columns
df["vehicle_id"] = df["vehicle_id"].astype(str)
df["vehicle_group"] = df["vehicle_group"].astype(str)
df["edge_id"] = df["edge_id"].astype(str)
df["lane_id"] = df["lane_id"].astype(str)

# vehicle_type is optional metadata, not required for LSTM
if "vehicle_type" in df.columns:
    df["vehicle_type"] = df["vehicle_type"].fillna("unknown").astype(str)

# Numeric columns required for model/training
numeric_cols = [
    "time",
    "x",
    "y",
    "speed_mps",
    "acceleration_mps2",
    "angle_deg",
    "lane_position_m",
    "distance_to_junction_center_m",
    "is_near_target_junction",
]

# Optional numeric metadata
optional_numeric_cols = [
    "target_junction_x",
    "target_junction_y",
]

for col in numeric_cols + optional_numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows only if required model columns are missing
required_cols = [
    "time",
    "vehicle_id",
    "vehicle_group",
    "x",
    "y",
    "speed_mps",
    "acceleration_mps2",
    "angle_deg",
    "lane_position_m",
    "distance_to_junction_center_m",
    "is_near_target_junction",
]

rows_before_dropna = len(df)
df = df.dropna(subset=required_cols).copy()
rows_after_dropna = len(df)

df = df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)

print("\nMissing values AFTER required-column cleaning:")
print(df.isna().sum())

print(f"\nRows before required-column dropna: {rows_before_dropna}")
print(f"Rows after required-column dropna : {rows_after_dropna}")
print(f"Rows removed                     : {rows_before_dropna - rows_after_dropna}")

print("\nColumn data types after cleaning:")
print(df.dtypes)

print("\nVehicle group counts:")
print(df["vehicle_group"].value_counts())

print("\nNear target junction counts:")
print(df["is_near_target_junction"].value_counts())

print("\nSpeed summary after SUMO-side filtering:")
print(df["speed_mps"].describe())

# =========================================================
# 5. Optional near-junction filtering for model data
# =========================================================
if USE_NEAR_JUNCTION_ONLY:                                      # use 1 herer to filter out the dataset 
    df = df[df["is_near_target_junction"] == 1].copy()
    filter_status = "ON - using only near-junction rows"
else:
    df = df.copy()
    filter_status = "OFF - using all dataset rows"

df = df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)

print(f"\nNear-junction model filter: {filter_status}")
print("Dataset shape after filter setting:", df.shape)
print("Unique vehicles after filter setting:", df["vehicle_id"].nunique())

num_vehicles_after_filter = df["vehicle_id"].nunique()

if num_vehicles_after_filter < 10:
    print("\nWARNING: Very few unique vehicles remain after filter setting.")
    print("LSTM train/validation/test split may fail or produce unreliable results.")

    if USE_NEAR_JUNCTION_ONLY:
        print("Consider increasing NEAR_JUNCTION_RADIUS_M or checking targeted routes.")
    else:
        print("Consider generating more vehicles or checking the SUMO route file.")

    print("\nNearest vehicles to target junction:")
    print(
        df.groupby("vehicle_id")["distance_to_junction_center_m"]
          .min()
          .sort_values()
          .head(20)
    )

    print("\nNumber of vehicles within distance thresholds:")
    for r in [50, 75, 100, 150, 200, 300]:
        temp = df[df["distance_to_junction_center_m"] <= r]
        print(
            f"Radius {r:>3} m: "
            f"vehicles={temp['vehicle_id'].nunique()}, rows={temp.shape[0]}"
        )

if df.empty:
    raise ValueError(
        "Dataset is empty after filter setting. "
        "Set USE_NEAR_JUNCTION_ONLY=False or regenerate the dataset with a larger near-junction radius."
    )

# =========================================================
# 6. Add angle sin/cos and velocity component features
# =========================================================
angle_rad = np.deg2rad(df["angle_deg"].astype(float))

df["angle_sin"] = np.sin(angle_rad)
df["angle_cos"] = np.cos(angle_rad)

# Velocity components from speed and heading angle
# SUMO angle convention: 0° = north/up, 90° = east/right
df["vx_mps"] = df["speed_mps"] * df["angle_sin"]
df["vy_mps"] = df["speed_mps"] * df["angle_cos"]

print("\nVelocity component summary:")
print(df[["speed_mps", "vx_mps", "vy_mps"]].describe())

# =========================================================
# 7. Check time step
# =========================================================
all_diffs = []

for vid, group in df.groupby("vehicle_id"):
    times = group["time"].values
    if len(times) > 1:
        diffs = np.diff(times)
        all_diffs.extend(np.round(diffs, 6).tolist())

if all_diffs:
    unique_diffs = sorted(set(all_diffs))
    print("\nDetected time step(s):", unique_diffs[:20])

    common_dt = pd.Series(all_diffs).mode().iloc[0]
    print(f"Most common detected time step: {common_dt}")

    if abs(common_dt - DT_SECONDS) > 1e-6:
        print(
            f"\nWARNING: Expected DT_SECONDS={DT_SECONDS}, "
            f"but most common dataset step is {common_dt}."
        )
else:
    print("\nWARNING: Could not detect time step because no vehicle has multiple records.")

# =========================================================
# 8. Define features and targets
# =========================================================
feature_cols = [
    "x",
    "y",
    "speed_mps",
    "vx_mps",
    "vy_mps",
    "acceleration_mps2",
    "angle_sin",
    "angle_cos",
    "lane_position_m",
    "distance_to_junction_center_m",
]
target_cols = ["x", "y"]

# These are constant for one intersection, so keep them as metadata only.
# Do NOT use them as LSTM features unless training across multiple intersections.
optional_meta_cols = [
    "target_junction_x",
    "target_junction_y",
]

# =========================================================
# 9. Train / validation / test split by vehicle_id
# =========================================================
vehicle_ids = df["vehicle_id"].unique()
num_vehicles = len(vehicle_ids)

if num_vehicles < 10:
    raise ValueError(
        f"Only {num_vehicles} unique vehicles are available after filtering. "
        "This is too few for a reliable train/validation/test split."
    )

# First: keep 10% of vehicles as final test set
train_pool_ids, test_ids = train_test_split(
    vehicle_ids,
    test_size=0.10,
    random_state=42
)

# Second: take 10% of the remaining 90% as validation
train_ids, val_ids = train_test_split(
    train_pool_ids,
    test_size=0.10,
    random_state=42
)

train_df = df[df["vehicle_id"].isin(train_ids)].copy()
val_df = df[df["vehicle_id"].isin(val_ids)].copy()
test_df = df[df["vehicle_id"].isin(test_ids)].copy()

print("\nUnique vehicles:")
print("Train     :", len(train_ids))
print("Validation:", len(val_ids))
print("Test      :", len(test_ids))

print("\nActual vehicle split percentages:")
print(f"Train     : {len(train_ids) / num_vehicles * 100:.2f}%")
print(f"Validation: {len(val_ids) / num_vehicles * 100:.2f}%")
print(f"Test      : {len(test_ids) / num_vehicles * 100:.2f}%")

print("\nApproximate final split:")
print("Train     : 81%")
print("Validation: 9%")
print("Test      : 10%")

print("\nRows:")
print("Train     :", train_df.shape[0])
print("Validation:", val_df.shape[0])
print("Test      :", test_df.shape[0])

# =========================================================
# 10. Scale data
# =========================================================
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit scalers only on training data
feature_scaler.fit(train_df[feature_cols])
target_scaler.fit(train_df[target_cols])

train_df_scaled = train_df.copy()
val_df_scaled = val_df.copy()
test_df_scaled = test_df.copy()

train_df_scaled[feature_cols] = feature_scaler.transform(train_df[feature_cols])
val_df_scaled[feature_cols] = feature_scaler.transform(val_df[feature_cols])
test_df_scaled[feature_cols] = feature_scaler.transform(test_df[feature_cols])

train_df_scaled[target_cols] = target_scaler.transform(train_df[target_cols])
val_df_scaled[target_cols] = target_scaler.transform(val_df[target_cols])
test_df_scaled[target_cols] = target_scaler.transform(test_df[target_cols])

# =========================================================
# 11. Create sequences
# =========================================================
def create_sequences(dataframe, input_len, pred_len, feature_cols, target_cols):
    X, y, meta = [], [], []
    skipped_noncontinuous = 0

    for vehicle_id, group in dataframe.groupby("vehicle_id"):
        group = group.sort_values("time").reset_index(drop=True)

        feature_data = group[feature_cols].values
        target_data = group[target_cols].values
        time_data = group["time"].values

        total_len = len(group)

        if total_len < input_len + pred_len:
            continue

        for i in range(
            0,
            total_len - input_len - pred_len + 1,
            SEQUENCE_STRIDE
        ):
            full_seq_times = time_data[i:i + input_len + pred_len]
            time_diffs = np.diff(full_seq_times)

            # Keep only continuous 0.1-second sequences
            if not np.allclose(time_diffs, DT_SECONDS, atol=1e-6):
                skipped_noncontinuous += 1
                continue

            x_seq = feature_data[i:i + input_len]
            y_seq = target_data[i + input_len:i + input_len + pred_len]

            pred_times = time_data[i + input_len:i + input_len + pred_len]

            X.append(x_seq)
            y.append(y_seq)

            meta_row = {
                "vehicle_id": vehicle_id,
                "vehicle_group": group.loc[i, "vehicle_group"],
                "input_start_time": time_data[i],
                "input_end_time": time_data[i + input_len - 1],
                "pred_start_time": pred_times[0],
                "pred_end_time": pred_times[-1],
                "pred_times": pred_times,
            }

            for col in optional_meta_cols:
                if col in group.columns:
                    meta_row[col] = group.loc[i, col]

            meta.append(meta_row)

    print(f"Skipped non-continuous sequences: {skipped_noncontinuous}")

    return np.array(X), np.array(y), meta


X_train, y_train, meta_train = create_sequences(
    train_df_scaled,
    INPUT_LEN,
    PRED_LEN,
    feature_cols,
    target_cols
)

X_val, y_val, meta_val = create_sequences(
    val_df_scaled,
    INPUT_LEN,
    PRED_LEN,
    feature_cols,
    target_cols
)

X_test, y_test, meta_test = create_sequences(
    test_df_scaled,
    INPUT_LEN,
    PRED_LEN,
    feature_cols,
    target_cols
)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape  :", X_val.shape)
print("y_val shape  :", y_val.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError(
        "No sequences were created for train/validation/test. "
        "Check near-junction filter, INPUT_LEN, PRED_LEN, and dataset length."
    )

y_train_flat = y_train.reshape((y_train.shape[0], y_train.shape[1] * y_train.shape[2]))
y_val_flat = y_val.reshape((y_val.shape[0], y_val.shape[1] * y_val.shape[2]))
y_test_flat = y_test.reshape((y_test.shape[0], y_test.shape[1] * y_test.shape[2]))

# =========================================================
# 12. Build LSTM model
# =========================================================
model = Sequential([
    Input(shape=(INPUT_LEN, len(feature_cols))),
    LSTM(128, return_sequences=False), #64
    Dropout(0.2), # 2
    Dense(64, activation="relu"),
    Dense(PRED_LEN * len(target_cols))
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

print("\nModel summary:")
model.summary()

# =========================================================
# 13. Train model
# =========================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train_flat,
    validation_data=(X_val, y_val_flat),
    epochs=20,
    batch_size=32,
    verbose=1,
    callbacks=[early_stop, checkpoint]
)

# =========================================================
# 13B. Save task 5 inference artifacts with the trained model
# =========================================================
joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
joblib.dump(target_scaler, TARGET_SCALER_PATH)

task5_metadata = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_path": project_relative_path(CSV_PATH),
    "model_path": project_relative_path(BEST_MODEL_PATH),
    "feature_scaler_path": project_relative_path(FEATURE_SCALER_PATH),
    "target_scaler_path": project_relative_path(TARGET_SCALER_PATH),
    "input_len": INPUT_LEN,
    "pred_len": PRED_LEN,
    "dt_seconds": DT_SECONDS,
    "sequence_stride": SEQUENCE_STRIDE,
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "use_near_junction_only": USE_NEAR_JUNCTION_ONLY,
    "intersection_id": "cluster_255722000_4115305935",
    "near_junction_radius_m": 100.0,
    "intersection_arrival_radius_m": 10.0,
    "collision_distance_m": 5.0,
    "high_time_diff_s": 2.0,
    "low_time_diff_s": 4.0,
    "high_ttc_s": 2.0,
    "low_ttc_s": 4.0,
    "training_rows": int(train_df.shape[0]),
    "validation_rows": int(val_df.shape[0]),
    "test_rows": int(test_df.shape[0]),
    "unique_train_vehicles": int(len(train_ids)),
    "unique_validation_vehicles": int(len(val_ids)),
    "unique_test_vehicles": int(len(test_ids)),
    "model_info": {
        "input_shape": [None, INPUT_LEN, len(feature_cols)],
        "output_units": PRED_LEN * len(target_cols),
        "inferred_input_len": INPUT_LEN,
        "inferred_feature_count": len(feature_cols),
        "inferred_pred_len": PRED_LEN,
    },
}

with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(task5_metadata, f, indent=2)

print("\nSaved task 5 inference artifacts:")
print(f"Feature scaler: {FEATURE_SCALER_PATH}")
print(f"Target scaler : {TARGET_SCALER_PATH}")
print(f"Metadata      : {MODEL_METADATA_PATH}")

# =========================================================
# 14. Predict on test set only for official evaluation
# =========================================================
y_pred_test_flat = model.predict(X_test, verbose=0)

y_pred_test = y_pred_test_flat.reshape((-1, PRED_LEN, len(target_cols)))
y_true_test = y_test_flat.reshape((-1, PRED_LEN, len(target_cols)))

y_pred_test_inv = target_scaler.inverse_transform(
    y_pred_test.reshape(-1, len(target_cols))
).reshape(y_pred_test.shape)

y_true_test_inv = target_scaler.inverse_transform(
    y_true_test.reshape(-1, len(target_cols))
).reshape(y_true_test.shape)

# =========================================================
# 15. Evaluate global model performance on test set only
# =========================================================
y_pred_2d = y_pred_test_inv.reshape(-1, 2)
y_true_2d = y_true_test_inv.reshape(-1, 2)

mse = mean_squared_error(y_true_2d, y_pred_2d)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_2d, y_pred_2d)

# X/Y accuracy based directly on predicted vs true coordinate values
x_abs_errors = np.abs(y_pred_2d[:, 0] - y_true_2d[:, 0])
y_abs_errors = np.abs(y_pred_2d[:, 1] - y_true_2d[:, 1])

x_accuracy_values = np.maximum(
    0.0,
    100.0 * (1.0 - x_abs_errors / np.maximum(np.abs(y_true_2d[:, 0]), 1e-6))
)

y_accuracy_values = np.maximum(
    0.0,
    100.0 * (1.0 - y_abs_errors / np.maximum(np.abs(y_true_2d[:, 1]), 1e-6))
)

xy_accuracy_values = (x_accuracy_values + y_accuracy_values) / 2.0

mean_x_accuracy = np.mean(x_accuracy_values)
mean_y_accuracy = np.mean(y_accuracy_values)
mean_xy_accuracy = np.mean(xy_accuracy_values)

print("\nTrajectory Prediction Results - Test Set Only")
print("MSE :", mse, "m²")
print("RMSE:", rmse, "m")
print("MAE :", mae, "m")
# print(f"Mean X Accuracy: {mean_x_accuracy:.2f}%")
# print(f"Mean Y Accuracy: {mean_y_accuracy:.2f}%")
# print(f"Mean Combined XY Accuracy: {mean_xy_accuracy:.2f}%")

# Optional relative accuracy
x_range = y_true_2d[:, 0].max() - y_true_2d[:, 0].min()
y_range = y_true_2d[:, 1].max() - y_true_2d[:, 1].min()
trajectory_range = np.sqrt(x_range**2 + y_range**2)

nrmse_percent = (rmse / trajectory_range) * 100 if trajectory_range != 0 else np.nan
relative_accuracy = 100 - nrmse_percent if trajectory_range != 0 else np.nan

print(f"Normalized RMSE: {nrmse_percent:.2f}%")
print(f"Prediction Accuracy: {relative_accuracy:.2f}%")

# # =========================================================
# # 15B. Evaluate TRAIN / VALIDATION / TEST accuracy
# # =========================================================

# def evaluate_dataset(model, X, y_flat, scaler, dataset_name="Dataset"):

#     y_pred_flat = model.predict(X, verbose=0)

#     y_pred = y_pred_flat.reshape((-1, PRED_LEN, len(target_cols)))
#     y_true = y_flat.reshape((-1, PRED_LEN, len(target_cols)))

#     y_pred_inv = scaler.inverse_transform(
#         y_pred.reshape(-1, len(target_cols))
#     ).reshape(y_pred.shape)

#     y_true_inv = scaler.inverse_transform(
#         y_true.reshape(-1, len(target_cols))
#     ).reshape(y_true.shape)

#     y_pred_2d = y_pred_inv.reshape(-1, 2)
#     y_true_2d = y_true_inv.reshape(-1, 2)

#     mse = mean_squared_error(y_true_2d, y_pred_2d)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true_2d, y_pred_2d)

#     # Normalized RMSE
#     x_range = y_true_2d[:, 0].max() - y_true_2d[:, 0].min()
#     y_range = y_true_2d[:, 1].max() - y_true_2d[:, 1].min()

#     trajectory_range = np.sqrt(x_range**2 + y_range**2)

#     nrmse_percent = (
#         (rmse / trajectory_range) * 100
#         if trajectory_range != 0 else np.nan
#     )

#     accuracy = (
#         100 - nrmse_percent
#         if trajectory_range != 0 else np.nan
#     )

#     print(f"\n===== {dataset_name} RESULTS =====")
#     print(f"MSE  : {mse:.4f}")
#     print(f"RMSE : {rmse:.4f} m")
#     print(f"MAE  : {mae:.4f} m")
#     print(f"Normalized RMSE : {nrmse_percent:.2f}%")
#     print(f"Prediction Accuracy : {accuracy:.2f}%")

#     return {
#         "mse": mse,
#         "rmse": rmse,
#         "mae": mae,
#         "nrmse_percent": nrmse_percent,
#         "accuracy": accuracy,
#     }


# # ---------------- TRAIN ----------------
# train_results = evaluate_dataset(
#     model,
#     X_train,
#     y_train_flat,
#     target_scaler,
#     dataset_name="TRAIN"
# )

# # ---------------- VALIDATION ----------------
# val_results = evaluate_dataset(
#     model,
#     X_val,
#     y_val_flat,
#     target_scaler,
#     dataset_name="VALIDATION"
# )

# # ---------------- TEST ----------------
# test_results = evaluate_dataset(
#     model,
#     X_test,
#     y_test_flat,
#     target_scaler,
#     dataset_name="TEST"
# )

# =========================================================
# 16. Show one sample prediction
# =========================================================
sample_idx = 0

print("\nSample meta information:")
sample_meta_print = meta_test[sample_idx].copy()
sample_meta_print.pop("pred_times", None)
print(sample_meta_print)

print("\nPredicted trajectory:")
print(y_pred_test_inv[sample_idx])

print("\nActual trajectory:")
print(y_true_test_inv[sample_idx])

# =========================================================
# 17. Plot training history
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches="tight")
print(f"\nSaved loss plot to:\n{LOSS_PLOT_PATH}")
plt.show()

# # =========================================================
# # 18. Optional: Plot one sample actual vs predicted trajectory
# # =========================================================
# plt.figure(figsize=(8, 6))
# plt.plot(
#     y_true_test_inv[sample_idx, :, 0],
#     y_true_test_inv[sample_idx, :, 1],
#     marker="o",
#     linewidth=2,
#     label="Actual"
# )
# plt.plot(
#     y_pred_test_inv[sample_idx, :, 0],
#     y_pred_test_inv[sample_idx, :, 1],
#     marker="x",
#     linewidth=2,
#     label="Predicted"
# )
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Actual vs Predicted Trajectory - Single Test Sample")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(SINGLE_TRAJ_PLOT_PATH, dpi=300, bbox_inches="tight")
# print(f"\nSaved single trajectory plot to:\n{SINGLE_TRAJ_PLOT_PATH}")
# plt.show()

# =========================================================
# 19. Save predictions with per-point metrics for TEST SET ONLY
# =========================================================
rows = []

for i in range(len(y_pred_test_inv)):
    info = meta_test[i]
    pred_times = info["pred_times"]

    for step in range(PRED_LEN):
        pred_x = float(y_pred_test_inv[i][step][0])
        pred_y = float(y_pred_test_inv[i][step][1])
        true_x = float(y_true_test_inv[i][step][0])
        true_y = float(y_true_test_inv[i][step][1])

        error_x = pred_x - true_x
        error_y = pred_y - true_y

        abs_error_x = abs(error_x)
        abs_error_y = abs(error_y)

        point_mse = (error_x**2 + error_y**2) / 2.0
        point_rmse = np.sqrt(point_mse)
        point_mae = (abs_error_x + abs_error_y) / 2.0

        x_accuracy = max(
            0.0,
            100.0 * (1.0 - abs_error_x / max(abs(true_x), 1e-6))
        )

        y_accuracy = max(
            0.0,
            100.0 * (1.0 - abs_error_y / max(abs(true_y), 1e-6))
        )

        xy_accuracy = (x_accuracy + y_accuracy) / 2.0

        row = {
            "data_split": "test",

            "vehicle_id": info["vehicle_id"],
            "vehicle_group": info["vehicle_group"],
            "window_start_time": round(float(info["input_start_time"]), 2),
            "window_end_time": round(float(info["input_end_time"]), 2),
            "pred_time": round(float(pred_times[step]), 2),

            "pred_x": round(pred_x, 4),
            "pred_y": round(pred_y, 4),
            "true_x": round(true_x, 4),
            "true_y": round(true_y, 4),

            "point_mse": round(float(point_mse), 6),
            "point_rmse": round(float(point_rmse), 6),
            "point_mae": round(float(point_mae), 6),

            "x_accuracy_percent": round(float(x_accuracy), 2),
            "y_accuracy_percent": round(float(y_accuracy), 2),
            "xy_accuracy_percent": round(float(xy_accuracy), 2),
        }

        for col in optional_meta_cols:
            if col in info:
                row[col] = info[col]

        rows.append(row)

pred_df = pd.DataFrame(rows)
pred_df.to_csv(OUTPUT_PRED_CSV, index=False)

print(f"\nSaved TEST SET predictions to:\n{OUTPUT_PRED_CSV}")
print(f"Best model saved to:\n{BEST_MODEL_PATH}")
print(f"Feature scaler saved to:\n{FEATURE_SCALER_PATH}")
print(f"Target scaler saved to:\n{TARGET_SCALER_PATH}")
print(f"Task 5 metadata saved to:\n{MODEL_METADATA_PATH}")

print("\nPrediction CSV columns:")
print(pred_df.columns.tolist())

print("\nPrediction rows:")
print(pred_df.shape[0])

print("\nUnique test vehicles:")
print(pred_df["vehicle_id"].nunique())

print("\nFirst 5 prediction rows:")
print(pred_df.head())

# =========================================================
# 20. Plot all vehicle trajectories in one plot
# =========================================================
# For visual clarity, plot only test predictions on top of test trajectories.
plot_pred_df = pred_df.copy()

plt.figure(figsize=(12, 10))

# Plot actual full trajectories from test_df
for vehicle_id, group in test_df.groupby("vehicle_id"):
    group = group.sort_values("time")
    plt.plot(
        group["x"],
        group["y"],
        linewidth=1,
        alpha=0.6
    )

# Overlay predicted future points from test predictions only
plt.scatter(
    plot_pred_df["pred_x"],
    plot_pred_df["pred_y"],
    s=15,
    alpha=0.5,
    label="Predicted Future Points"
)

# Add target junction star
if "target_junction_x" in test_df.columns and "target_junction_y" in test_df.columns:
    jx = test_df["target_junction_x"].iloc[0]
    jy = test_df["target_junction_y"].iloc[0]

    plt.scatter(
        jx,
        jy,
        s=180,
        marker="*",
        label="Target intersection",
        zorder=5
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title(
    f"All Test Vehicle Trajectories with Predicted Future Points\n"
    f"Input: {INPUT_LEN * DT_SECONDS:.1f}s | "
    f"Prediction: {PRED_LEN * DT_SECONDS:.1f}s | "
    f"Stride: {SEQUENCE_STRIDE * DT_SECONDS:.1f}s"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(ALL_TRAJ_PLOT_PATH, dpi=300, bbox_inches="tight")
print(f"\nSaved all-vehicle trajectory plot to:\n{ALL_TRAJ_PLOT_PATH}")
plt.show()
