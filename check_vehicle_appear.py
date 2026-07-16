import pandas as pd
import numpy as np

df = pd.read_csv("vehicle_distance_to_intersection.csv")
df["vehicle_id"] = df["vehicle_id"].astype(str)
df = df.sort_values(["vehicle_id", "time"])

for vid in df["vehicle_id"].unique():
    times = df[df["vehicle_id"] == vid]["time"].values
    diffs = np.diff(times)
    print(vid, set(diffs))