from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots" / "intersection_arrival"

PREDICTIONS_CSV = DATA_DIR / "trajectory_predictions.csv"

OUTPUT_CONFLICT_CSV = DATA_DIR / "intersection_arrival_conflicts.csv"
OUTPUT_HIGH_RISK_CSV = DATA_DIR / "high_risk_intersection_arrival_events.csv"
OUTPUT_EPISODES_CSV = DATA_DIR / "intersection_arrival_risk_episodes.csv"

RISK_COUNT_PLOT = PLOTS_DIR / "intersection_arrival_risk_counts.png"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Used only to decide whether a vehicle actually arrives at the intersection
INTERSECTION_ARRIVAL_RADIUS_M = 10.0

# Risk rules
HIGH_TIME_DIFF_S = 2.0
LOW_TIME_DIFF_S = 4.0


def classify_arrival_risk(time_difference_s):
    if pd.isna(time_difference_s):
        return "SAFE"

    if time_difference_s <= HIGH_TIME_DIFF_S:
        return "HIGH"

    elif time_difference_s <= LOW_TIME_DIFF_S:
        return "LOW"

    else:
        return "SAFE"


def estimate_arrival_to_intersection(vdf, x_col="pred_x", y_col="pred_y"):
    """
    Estimate vehicle arrival time at the intersection.

    For predicted arrival:
        x_col='pred_x', y_col='pred_y'

    For real/actual arrival:
        x_col='true_x', y_col='true_y'
    """

    vdf = vdf.sort_values("pred_time").reset_index(drop=True)

    jx = float(vdf["target_junction_x"].iloc[0])
    jy = float(vdf["target_junction_y"].iloc[0])

    xs = vdf[x_col].to_numpy(dtype=float)
    ys = vdf[y_col].to_numpy(dtype=float)
    times = vdf["pred_time"].to_numpy(dtype=float)

    distances = np.sqrt((xs - jx) ** 2 + (ys - jy) ** 2)

    if len(distances) == 0:
        return None, None, np.nan, np.nan

    min_idx = int(np.argmin(distances))
    min_distance = float(distances[min_idx])
    closest_time = float(times[min_idx])

    # First time vehicle enters intersection radius = arrival time
    for i, d in enumerate(distances):
        if d <= INTERSECTION_ARRIVAL_RADIUS_M:
            return float(times[i]), int(i + 1), min_distance, closest_time

    # Vehicle did not arrive during prediction window
    return None, None, min_distance, closest_time


def build_vehicle_arrivals(group_df):
    arrivals = {}

    for vehicle_id, vdf in group_df.groupby("vehicle_id"):

        pred_arrival_time, pred_arrival_step, pred_min_dist, pred_closest_time = (
            estimate_arrival_to_intersection(vdf, "pred_x", "pred_y")
        )

        real_arrival_time, real_arrival_step, real_min_dist, real_closest_time = (
            estimate_arrival_to_intersection(vdf, "true_x", "true_y")
        )

        arrivals[str(vehicle_id)] = {
            "vehicle_group": vdf["vehicle_group"].iloc[0] if "vehicle_group" in vdf.columns else "unknown",

            # Predicted arrival information
            "arrival_time_s": pred_arrival_time,
            "arrival_step": pred_arrival_step,
            "min_distance_to_intersection_m": pred_min_dist,
            "closest_time_to_intersection_s": pred_closest_time,

            # Real/actual arrival information
            "real_arrival_time_s": real_arrival_time,
            "real_arrival_step": real_arrival_step,
            "real_min_distance_to_intersection_m": real_min_dist,
            "real_closest_time_to_intersection_s": real_closest_time,
        }

    return arrivals


def build_risk_episodes(conflict_df):
    if conflict_df.empty:
        return pd.DataFrame()

    df = conflict_df[conflict_df["arrival_risk_level"].isin(["HIGH", "LOW"])].copy()

    if df.empty:
        return pd.DataFrame()

    unique_times = np.sort(df["window_end_time"].dropna().unique())

    if len(unique_times) > 1:
        diffs = np.diff(unique_times)
        window_stride = float(pd.Series(np.round(diffs, 6)).mode().iloc[0])
    else:
        window_stride = 1.0

    tolerance = 1e-6

    df["pair_key"] = df.apply(
        lambda row: tuple(sorted([str(row["vehicle_1"]), str(row["vehicle_2"])])),
        axis=1
    )

    df = df.sort_values(["pair_key", "window_end_time"]).reset_index(drop=True)

    risk_priority = {"LOW": 1, "HIGH": 2}
    rows = []

    for pair_key, group in df.groupby("pair_key"):
        group = group.sort_values("window_end_time").reset_index(drop=True)
        current_episode = None

        for _, row in group.iterrows():
            current_time = float(row["window_end_time"])
            current_risk = row["arrival_risk_level"]

            if current_episode is None:
                current_episode = {
                    "vehicle_1": pair_key[0],
                    "vehicle_2": pair_key[1],
                    "episode_start_time": current_time,
                    "episode_end_time": current_time,
                    "episode_length_windows": 1,
                    "max_risk_level": current_risk,
                    "min_arrival_time_difference_s": row["arrival_time_difference_s"],
                    "min_real_arrival_time_difference_s": row["real_arrival_time_difference_s"],
                    "min_vehicle_1_distance_to_intersection_m": row["vehicle_1_min_distance_to_intersection_m"],
                    "min_vehicle_2_distance_to_intersection_m": row["vehicle_2_min_distance_to_intersection_m"],
                }
                continue

            expected_next_time = current_episode["episode_end_time"] + window_stride

            if abs(current_time - expected_next_time) <= tolerance:
                current_episode["episode_end_time"] = current_time
                current_episode["episode_length_windows"] += 1

                if risk_priority[current_risk] > risk_priority[current_episode["max_risk_level"]]:
                    current_episode["max_risk_level"] = current_risk

                if pd.notna(row["arrival_time_difference_s"]):
                    if (
                        pd.isna(current_episode["min_arrival_time_difference_s"])
                        or row["arrival_time_difference_s"] < current_episode["min_arrival_time_difference_s"]
                    ):
                        current_episode["min_arrival_time_difference_s"] = row["arrival_time_difference_s"]

                if pd.notna(row["real_arrival_time_difference_s"]):
                    if (
                        pd.isna(current_episode["min_real_arrival_time_difference_s"])
                        or row["real_arrival_time_difference_s"] < current_episode["min_real_arrival_time_difference_s"]
                    ):
                        current_episode["min_real_arrival_time_difference_s"] = row["real_arrival_time_difference_s"]

                current_episode["min_vehicle_1_distance_to_intersection_m"] = min(
                    current_episode["min_vehicle_1_distance_to_intersection_m"],
                    row["vehicle_1_min_distance_to_intersection_m"],
                )

                current_episode["min_vehicle_2_distance_to_intersection_m"] = min(
                    current_episode["min_vehicle_2_distance_to_intersection_m"],
                    row["vehicle_2_min_distance_to_intersection_m"],
                )

            else:
                rows.append(current_episode)

                current_episode = {
                    "vehicle_1": pair_key[0],
                    "vehicle_2": pair_key[1],
                    "episode_start_time": current_time,
                    "episode_end_time": current_time,
                    "episode_length_windows": 1,
                    "max_risk_level": current_risk,
                    "min_arrival_time_difference_s": row["arrival_time_difference_s"],
                    "min_real_arrival_time_difference_s": row["real_arrival_time_difference_s"],
                    "min_vehicle_1_distance_to_intersection_m": row["vehicle_1_min_distance_to_intersection_m"],
                    "min_vehicle_2_distance_to_intersection_m": row["vehicle_2_min_distance_to_intersection_m"],
                }

        if current_episode is not None:
            rows.append(current_episode)

    episodes_df = pd.DataFrame(rows)

    if not episodes_df.empty:
        episodes_df["episode_duration_s"] = (
            episodes_df["episode_end_time"] - episodes_df["episode_start_time"]
        ) + window_stride

    return episodes_df


def save_risk_count_plot(conflict_df, out_path):
    risk_order = ["SAFE", "LOW", "HIGH"]

    counts = (
        conflict_df["arrival_risk_level"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
    )

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)

    plt.xlabel("Arrival-Time Risk Level", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.title("Intersection Arrival-Time Risk Counts", fontsize=16)

    plt.yscale("log")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Load trajectory predictions
# =========================================================
if not PREDICTIONS_CSV.exists():
    raise FileNotFoundError(f"Prediction file not found:\n{PREDICTIONS_CSV}")

df = pd.read_csv(PREDICTIONS_CSV)

required_cols = {
    "vehicle_id",
    "vehicle_group",
    "window_start_time",
    "window_end_time",
    "pred_time",
    "pred_x",
    "pred_y",
    "true_x",
    "true_y",
    "target_junction_x",
    "target_junction_y",
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in prediction file: {missing}")

df["vehicle_id"] = df["vehicle_id"].astype(str)
df["vehicle_group"] = df["vehicle_group"].astype(str)

numeric_cols = [
    "window_start_time",
    "window_end_time",
    "pred_time",
    "pred_x",
    "pred_y",
    "true_x",
    "true_y",
    "target_junction_x",
    "target_junction_y",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

rows_before = len(df)
df = df.dropna(subset=numeric_cols).copy()
rows_after = len(df)

df = df.sort_values(
    ["window_start_time", "window_end_time", "vehicle_id", "pred_time"]
).reset_index(drop=True)

print("Predictions shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

print(f"\nRows before cleaning: {rows_before}")
print(f"Rows after cleaning : {rows_after}")
print(f"Rows removed        : {rows_before - rows_after}")


# =========================================================
# Compute intersection arrival-time conflicts
# =========================================================
rows = []

for (window_start_time, window_end_time), group in df.groupby(["window_start_time", "window_end_time"]):
    arrivals = build_vehicle_arrivals(group)
    vehicle_ids = sorted(arrivals.keys())

    if len(vehicle_ids) < 2:
        continue

    for veh_a, veh_b in combinations(vehicle_ids, 2):
        a = arrivals[veh_a]
        b = arrivals[veh_b]

        arrival_a = a["arrival_time_s"]
        arrival_b = b["arrival_time_s"]

        if arrival_a is not None and arrival_b is not None:
            arrival_diff = abs(arrival_a - arrival_b)
        else:
            arrival_diff = np.nan

        real_arrival_a = a["real_arrival_time_s"]
        real_arrival_b = b["real_arrival_time_s"]

        if real_arrival_a is not None and real_arrival_b is not None:
            real_arrival_diff = abs(real_arrival_a - real_arrival_b)
        else:
            real_arrival_diff = np.nan

        risk = classify_arrival_risk(arrival_diff)

        rows.append({
            "window_start_time": float(window_start_time),
            "window_end_time": float(window_end_time),

            "vehicle_1": veh_a,
            "vehicle_2": veh_b,

            "vehicle_1_group": a["vehicle_group"],
            "vehicle_2_group": b["vehicle_group"],

            # Predicted arrival information
            "vehicle_1_arrival_time_s": arrival_a,
            "vehicle_2_arrival_time_s": arrival_b,
            "arrival_time_difference_s": None if pd.isna(arrival_diff) else round(float(arrival_diff), 4),

            "vehicle_1_arrival_step": a["arrival_step"],
            "vehicle_2_arrival_step": b["arrival_step"],

            "vehicle_1_min_distance_to_intersection_m": round(a["min_distance_to_intersection_m"], 4),
            "vehicle_2_min_distance_to_intersection_m": round(b["min_distance_to_intersection_m"], 4),

            # Real/actual arrival information
            "vehicle_1_real_arrival_time_s": real_arrival_a,
            "vehicle_2_real_arrival_time_s": real_arrival_b,
            "real_arrival_time_difference_s": None if pd.isna(real_arrival_diff) else round(float(real_arrival_diff), 4),

            "vehicle_1_real_arrival_step": a["real_arrival_step"],
            "vehicle_2_real_arrival_step": b["real_arrival_step"],

            "vehicle_1_real_min_distance_to_intersection_m": round(a["real_min_distance_to_intersection_m"], 4),
            "vehicle_2_real_min_distance_to_intersection_m": round(b["real_min_distance_to_intersection_m"], 4),

            "arrival_risk_level": risk,
        })

conflict_df = pd.DataFrame(rows)


# =========================================================
# Save outputs
# =========================================================
if conflict_df.empty:
    print("\nNo intersection arrival conflicts were produced.")

else:
    risk_order = {"HIGH": 0, "LOW": 1, "SAFE": 2}
    conflict_df["risk_sort"] = conflict_df["arrival_risk_level"].map(risk_order).fillna(3)

    conflict_df = conflict_df.sort_values(
        by=["risk_sort", "arrival_time_difference_s", "window_end_time"],
        ascending=[True, True, True]
    ).drop(columns=["risk_sort"]).reset_index(drop=True)

    conflict_df.to_csv(OUTPUT_CONFLICT_CSV, index=False)

    high_df = conflict_df[conflict_df["arrival_risk_level"] == "HIGH"].copy()
    high_df.to_csv(OUTPUT_HIGH_RISK_CSV, index=False)

    episodes_df = build_risk_episodes(conflict_df)

    if not episodes_df.empty:
        episodes_df = episodes_df.sort_values(
            by=["episode_start_time", "max_risk_level"],
            ascending=[True, False]
        ).reset_index(drop=True)

        episodes_df.to_csv(OUTPUT_EPISODES_CSV, index=False)

    save_risk_count_plot(conflict_df, RISK_COUNT_PLOT)

    print("\nIntersection arrival conflicts saved to:")
    print(OUTPUT_CONFLICT_CSV)

    print("\nHigh-risk intersection arrival events saved to:")
    print(OUTPUT_HIGH_RISK_CSV)

    if not episodes_df.empty:
        print("\nIntersection arrival risk episodes saved to:")
        print(OUTPUT_EPISODES_CSV)

    print("\nRisk count plot saved to:")
    print(RISK_COUNT_PLOT)

    print("\nTotal vehicle-pair windows:", len(conflict_df))

    print("\nArrival risk counts:")
    print(
        conflict_df["arrival_risk_level"]
        .value_counts()
        .reindex(["SAFE", "LOW", "HIGH"], fill_value=0)
    )

    if not episodes_df.empty:
        print("\nEpisode risk counts:")
        print(episodes_df["max_risk_level"].value_counts())

        print("\nTop 10 risk episodes:")
        print(episodes_df.head(10))

    print("\nTop 10 most critical intersection-arrival conflicts:")
    print(
        conflict_df[
            [
                "window_start_time",
                "window_end_time",
                "vehicle_1",
                "vehicle_2",
                "vehicle_1_arrival_time_s",
                "vehicle_2_arrival_time_s",
                "arrival_time_difference_s",
                "vehicle_1_real_arrival_time_s",
                "vehicle_2_real_arrival_time_s",
                "real_arrival_time_difference_s",
                "arrival_risk_level",
            ]
        ].head(10)
    )
