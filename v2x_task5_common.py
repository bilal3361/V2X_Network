from __future__ import annotations

import json
import math
import zipfile
from collections import deque
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

TRAINING_DATASET_PATH = DATA_DIR / "vehicle_trajectory_dataset.csv"
MODEL_PATH = MODEL_DIR / "best_trajectory_model.keras"
FEATURE_SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = MODEL_DIR / "target_scaler.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "task5_model_metadata.json"

DT_SECONDS = 0.1
INPUT_LEN = 20
PRED_LEN = 30
SEQUENCE_STRIDE = 2
USE_NEAR_JUNCTION_ONLY = True

INTERSECTION_ID = "cluster_255722000_4115305935"
NEAR_JUNCTION_RADIUS_M = 100.0
INTERSECTION_ARRIVAL_RADIUS_M = 10.0

HIGH_TIME_DIFF_S = 2.0
LOW_TIME_DIFF_S = 4.0

FEATURE_COLS = [
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

TARGET_COLS = ["x", "y"]

REQUIRED_RAW_COLS = [
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

NUMERIC_RAW_COLS = [
    "time",
    "x",
    "y",
    "speed_mps",
    "acceleration_mps2",
    "angle_deg",
    "lane_position_m",
    "distance_to_junction_center_m",
    "is_near_target_junction",
    "target_junction_x",
    "target_junction_y",
]

RISK_PRIORITY = {"SAFE": 0, "LOW": 1, "HIGH": 2}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def infer_vehicle_group(vehicle_id: str) -> str:
    return "targeted" if str(vehicle_id).startswith("targeted_") else "background"


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    angle_rad = np.deg2rad(df["angle_deg"].astype(float))
    df["angle_sin"] = np.sin(angle_rad)
    df["angle_cos"] = np.cos(angle_rad)
    df["vx_mps"] = df["speed_mps"].astype(float) * df["angle_sin"]
    df["vy_mps"] = df["speed_mps"].astype(float) * df["angle_cos"]
    return df


def load_training_dataframe(
    csv_path: Path = TRAINING_DATASET_PATH,
    use_near_junction_only: bool = USE_NEAR_JUNCTION_ONLY,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {sorted(missing)}")

    df["vehicle_id"] = df["vehicle_id"].astype(str)
    df["vehicle_group"] = df["vehicle_group"].astype(str)

    for col in NUMERIC_RAW_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_RAW_COLS).copy()
    if df.empty:
        raise ValueError(f"No valid rows remain after cleaning {csv_path}")

    if use_near_junction_only:
        df = df[df["is_near_target_junction"] == 1].copy()

    if df.empty:
        raise ValueError("No rows remain after near-junction filtering.")

    df = df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)
    df = add_model_features(df)
    df.attrs["rows_removed_by_cleaning"] = before - len(df)
    return df


def split_training_vehicles(df: pd.DataFrame, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vehicle_ids = df["vehicle_id"].unique()
    if len(vehicle_ids) < 10:
        raise ValueError(f"Only {len(vehicle_ids)} vehicles are available; need at least 10.")

    train_pool_ids, test_ids = train_test_split(
        vehicle_ids,
        test_size=0.10,
        random_state=random_state,
    )
    train_ids, val_ids = train_test_split(
        train_pool_ids,
        test_size=0.10,
        random_state=random_state,
    )
    return train_ids, val_ids, test_ids


def inspect_keras_model_shape(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with zipfile.ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))

    layers = config.get("config", {}).get("layers", [])
    input_shape = None
    output_units = None

    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            input_shape = layer.get("config", {}).get("batch_shape")

    for layer in reversed(layers):
        units = layer.get("config", {}).get("units")
        if units is not None:
            output_units = int(units)
            break

    inferred_pred_len = None
    if output_units is not None and output_units % len(TARGET_COLS) == 0:
        inferred_pred_len = output_units // len(TARGET_COLS)

    return {
        "input_shape": input_shape,
        "output_units": output_units,
        "inferred_input_len": input_shape[1] if input_shape and len(input_shape) > 1 else None,
        "inferred_feature_count": input_shape[2] if input_shape and len(input_shape) > 2 else None,
        "inferred_pred_len": inferred_pred_len,
    }


def validate_model_shape(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    model_info = inspect_keras_model_shape(model_path)
    expected = {
        "inferred_input_len": INPUT_LEN,
        "inferred_feature_count": len(FEATURE_COLS),
        "inferred_pred_len": PRED_LEN,
    }

    mismatches = {
        key: {"expected": value, "actual": model_info.get(key)}
        for key, value in expected.items()
        if model_info.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Model shape does not match task 5 metadata: {mismatches}")

    return model_info


def export_scalers_and_metadata(
    dataset_path: Path = TRAINING_DATASET_PATH,
    model_path: Path = MODEL_PATH,
    feature_scaler_path: Path = FEATURE_SCALER_PATH,
    target_scaler_path: Path = TARGET_SCALER_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> dict[str, Any]:
    model_info = validate_model_shape(model_path)
    df = load_training_dataframe(dataset_path)
    train_ids, val_ids, test_ids = split_training_vehicles(df)
    train_df = df[df["vehicle_id"].isin(train_ids)].copy()

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLS])
    target_scaler.fit(train_df[TARGET_COLS])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    metadata = {
        "created_at_utc": utc_now_iso(),
        "dataset_path": project_relative_path(dataset_path),
        "model_path": project_relative_path(model_path),
        "feature_scaler_path": project_relative_path(feature_scaler_path),
        "target_scaler_path": project_relative_path(target_scaler_path),
        "input_len": INPUT_LEN,
        "pred_len": PRED_LEN,
        "dt_seconds": DT_SECONDS,
        "sequence_stride": SEQUENCE_STRIDE,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "use_near_junction_only": USE_NEAR_JUNCTION_ONLY,
        "intersection_id": INTERSECTION_ID,
        "near_junction_radius_m": NEAR_JUNCTION_RADIUS_M,
        "intersection_arrival_radius_m": INTERSECTION_ARRIVAL_RADIUS_M,
        "high_time_diff_s": HIGH_TIME_DIFF_S,
        "low_time_diff_s": LOW_TIME_DIFF_S,
        "model_info": model_info,
        "training_rows": int(len(train_df)),
        "unique_train_vehicles": int(len(train_ids)),
        "unique_validation_vehicles": int(len(val_ids)),
        "unique_test_vehicles": int(len(test_ids)),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_metadata(metadata_path: Path = MODEL_METADATA_PATH) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def ensure_artifacts_exist(
    model_path: Path = MODEL_PATH,
    feature_scaler_path: Path = FEATURE_SCALER_PATH,
    target_scaler_path: Path = TARGET_SCALER_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> None:
    missing = [
        path
        for path in [model_path, feature_scaler_path, target_scaler_path, metadata_path]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Task 5 inference artifacts are missing. Run:\n"
            "  python scripts/export_task5_artifacts.py\n\n"
            f"Missing files:\n{missing_text}"
        )


def project_relative_path(path: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_model_feature_row(state: dict[str, Any]) -> dict[str, float]:
    angle_rad = math.radians(float(state["angle_deg"]))
    speed = float(state["speed_mps"])
    return {
        "x": float(state["x"]),
        "y": float(state["y"]),
        "speed_mps": speed,
        "vx_mps": speed * math.sin(angle_rad),
        "vy_mps": speed * math.cos(angle_rad),
        "acceleration_mps2": float(state["acceleration_mps2"]),
        "angle_sin": math.sin(angle_rad),
        "angle_cos": math.cos(angle_rad),
        "lane_position_m": float(state["lane_position_m"]),
        "distance_to_junction_center_m": float(state["distance_to_junction_center_m"]),
    }


class VehicleHistoryStore:
    def __init__(self, input_len: int = INPUT_LEN) -> None:
        self.input_len = input_len
        self._buffers: dict[str, deque[dict[str, Any]]] = {}

    def add(self, vehicle_id: str, state: dict[str, Any]) -> None:
        buffer = self._buffers.setdefault(str(vehicle_id), deque(maxlen=self.input_len))
        buffer.append(state)

    def is_ready(self, vehicle_id: str) -> bool:
        return len(self._buffers.get(str(vehicle_id), [])) >= self.input_len

    def history(self, vehicle_id: str) -> list[dict[str, Any]]:
        return list(self._buffers.get(str(vehicle_id), []))

    def prune_missing(self, active_vehicle_ids: set[str]) -> None:
        for vehicle_id in list(self._buffers):
            if vehicle_id not in active_vehicle_ids:
                del self._buffers[vehicle_id]


class TrajectoryPredictor:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_scaler_path: Path = FEATURE_SCALER_PATH,
        target_scaler_path: Path = TARGET_SCALER_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
    ) -> None:
        ensure_artifacts_exist(model_path, feature_scaler_path, target_scaler_path, metadata_path)
        self.metadata = load_metadata(metadata_path)
        self.input_len = int(self.metadata["input_len"])
        self.pred_len = int(self.metadata["pred_len"])
        self.dt_seconds = float(self.metadata["dt_seconds"])
        self.feature_cols = list(self.metadata["feature_cols"])
        self.target_cols = list(self.metadata["target_cols"])
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)

        try:
            from tensorflow.keras.models import load_model
        except Exception as exc:
            raise RuntimeError("TensorFlow/Keras is required to load the trajectory model.") from exc

        self.model = load_model(model_path)

    def predict_vehicle(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(history) < self.input_len:
            raise ValueError(f"Need {self.input_len} history rows, got {len(history)}")

        window = history[-self.input_len :]
        feature_rows = [build_model_feature_row(row) for row in window]
        feature_df = pd.DataFrame(feature_rows, columns=self.feature_cols)
        scaled = self.feature_scaler.transform(feature_df)
        model_input = scaled.reshape(1, self.input_len, len(self.feature_cols))

        pred_scaled = self.model.predict(model_input, verbose=0)
        pred_scaled = pred_scaled.reshape(self.pred_len, len(self.target_cols))
        pred_xy = self.target_scaler.inverse_transform(pred_scaled)

        last = window[-1]
        base_time = float(last["time"])
        vehicle_id = str(last["vehicle_id"])
        vehicle_group = str(last.get("vehicle_group", infer_vehicle_group(vehicle_id)))

        rows = []
        for idx, xy in enumerate(pred_xy):
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "vehicle_group": vehicle_group,
                    "pred_step": idx + 1,
                    "pred_time": round(base_time + ((idx + 1) * self.dt_seconds), 4),
                    "pred_x": float(xy[0]),
                    "pred_y": float(xy[1]),
                    "target_junction_x": float(last["target_junction_x"]),
                    "target_junction_y": float(last["target_junction_y"]),
                }
            )
        return rows

    def predict_many(
        self,
        histories_by_vehicle: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        ready_items = [
            (str(vehicle_id), history[-self.input_len :])
            for vehicle_id, history in histories_by_vehicle.items()
            if len(history) >= self.input_len
        ]

        if not ready_items:
            return {}

        all_feature_rows = []
        last_rows = []

        for _, window in ready_items:
            all_feature_rows.extend(build_model_feature_row(row) for row in window)
            last_rows.append(window[-1])

        feature_df = pd.DataFrame(all_feature_rows, columns=self.feature_cols)
        scaled = self.feature_scaler.transform(feature_df)
        model_input = scaled.reshape(len(ready_items), self.input_len, len(self.feature_cols))

        pred_scaled = self.model.predict(model_input, verbose=0)
        pred_scaled = pred_scaled.reshape(len(ready_items), self.pred_len, len(self.target_cols))
        pred_xy = self.target_scaler.inverse_transform(
            pred_scaled.reshape(-1, len(self.target_cols))
        ).reshape(len(ready_items), self.pred_len, len(self.target_cols))

        predictions_by_vehicle: dict[str, list[dict[str, Any]]] = {}

        for vehicle_index, (vehicle_id, _) in enumerate(ready_items):
            last = last_rows[vehicle_index]
            base_time = float(last["time"])
            vehicle_group = str(last.get("vehicle_group", infer_vehicle_group(vehicle_id)))
            rows = []

            for idx, xy in enumerate(pred_xy[vehicle_index]):
                rows.append(
                    {
                        "vehicle_id": vehicle_id,
                        "vehicle_group": vehicle_group,
                        "pred_step": idx + 1,
                        "pred_time": round(base_time + ((idx + 1) * self.dt_seconds), 4),
                        "pred_x": float(xy[0]),
                        "pred_y": float(xy[1]),
                        "target_junction_x": float(last["target_junction_x"]),
                        "target_junction_y": float(last["target_junction_y"]),
                    }
                )

            predictions_by_vehicle[vehicle_id] = rows

        return predictions_by_vehicle


def classify_arrival_risk(time_difference_s: float | None) -> str:
    if time_difference_s is None or pd.isna(time_difference_s):
        return "SAFE"
    if time_difference_s <= HIGH_TIME_DIFF_S:
        return "HIGH"
    if time_difference_s <= LOW_TIME_DIFF_S:
        return "LOW"
    return "SAFE"


def estimate_arrival_to_intersection(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "arrival_time_s": None,
            "arrival_step": None,
            "min_distance_to_intersection_m": None,
            "closest_time_to_intersection_s": None,
        }

    jx = float(predictions[0]["target_junction_x"])
    jy = float(predictions[0]["target_junction_y"])

    distances = [
        euclidean_distance(float(row["pred_x"]), float(row["pred_y"]), jx, jy)
        for row in predictions
    ]
    min_idx = int(np.argmin(distances))

    for idx, distance in enumerate(distances):
        if distance <= INTERSECTION_ARRIVAL_RADIUS_M:
            return {
                "arrival_time_s": float(predictions[idx]["pred_time"]),
                "arrival_step": int(idx + 1),
                "min_distance_to_intersection_m": float(distances[min_idx]),
                "closest_time_to_intersection_s": float(predictions[min_idx]["pred_time"]),
            }

    return {
        "arrival_time_s": None,
        "arrival_step": None,
        "min_distance_to_intersection_m": float(distances[min_idx]),
        "closest_time_to_intersection_s": float(predictions[min_idx]["pred_time"]),
    }


def build_pair_alerts(
    predictions_by_vehicle: dict[str, list[dict[str, Any]]],
    simulation_time: float,
    min_risk_level: str = "LOW",
) -> list[dict[str, Any]]:
    """Build MQTT alert payloads using arrival-time difference only.

    Risk rule:
        arrival_time_difference_s <= HIGH_TIME_DIFF_S -> HIGH
        arrival_time_difference_s <= LOW_TIME_DIFF_S  -> LOW
        otherwise                                    -> SAFE
    """
    vehicle_ids = sorted(predictions_by_vehicle)
    if len(vehicle_ids) < 2:
        return []

    arrivals = {
        vehicle_id: estimate_arrival_to_intersection(predictions_by_vehicle[vehicle_id])
        for vehicle_id in vehicle_ids
    }

    alerts = []
    min_priority = RISK_PRIORITY[min_risk_level]

    for vehicle_1, vehicle_2 in combinations(vehicle_ids, 2):
        pred_1 = predictions_by_vehicle[vehicle_1]
        pred_2 = predictions_by_vehicle[vehicle_2]
        arrival_1 = arrivals[vehicle_1]
        arrival_2 = arrivals[vehicle_2]

        arrival_diff = None
        if arrival_1["arrival_time_s"] is not None and arrival_2["arrival_time_s"] is not None:
            arrival_diff = abs(float(arrival_1["arrival_time_s"]) - float(arrival_2["arrival_time_s"]))

        arrival_risk = classify_arrival_risk(arrival_diff)
        risk_level = arrival_risk

        if RISK_PRIORITY[risk_level] < min_priority:
            continue

        group_1 = pred_1[0].get("vehicle_group", infer_vehicle_group(vehicle_1))
        group_2 = pred_2[0].get("vehicle_group", infer_vehicle_group(vehicle_2))
        pair_id = f"{vehicle_1}|{vehicle_2}"
        recommendation = (
            "Issue immediate V2X intersection-arrival warning"
            if risk_level == "HIGH"
            else "Monitor arrival timing and prepare warning if risk escalates"
        )

        alerts.append(
            {
                "alert_id": f"arrival:{pair_id}:{round(float(simulation_time), 1)}:{risk_level}",
                "pair_id": pair_id,
                "event_type": "v2x_intersection_arrival_risk",
                "generated_at_utc": utc_now_iso(),
                "simulation_time": round(float(simulation_time), 4),
                "risk_level": risk_level,
                "severity_rank": RISK_PRIORITY[risk_level],
                "vehicle_1": vehicle_1,
                "vehicle_2": vehicle_2,
                "vehicle_1_group": group_1,
                "vehicle_2_group": group_2,
                "arrival_risk_level": arrival_risk,
                "risk_source": "arrival",
                "arrival_time_difference_s": None if arrival_diff is None else round(float(arrival_diff), 4),
                "vehicle_1_arrival_time_s": arrival_1["arrival_time_s"],
                "vehicle_2_arrival_time_s": arrival_2["arrival_time_s"],
                "prediction_horizon_s": round(PRED_LEN * DT_SECONDS, 4),
                "recommendation": recommendation,
                "message": f"{risk_level} intersection arrival-time risk detected between {vehicle_1} and {vehicle_2}",
            }
        )

    return alerts

def risk_topic_suffix(risk_level: str) -> str:
    return str(risk_level).lower()
