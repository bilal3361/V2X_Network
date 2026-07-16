import sys
import math
import csv
from pathlib import Path

import traci

SUMO_BINARY = "sumo-gui"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_CONFIG = PROJECT_ROOT / "osm.sumocfg"
OUTPUT_CSV = PROJECT_ROOT / "data" / "vehicle_trajectory_dataset.csv"

# Selected main intersection for Scenario_1
INTERSECTION_ID = "cluster_255722000_4115305935"

# Keep all vehicles, but mark those close to the chosen intersection.
NEAR_JUNCTION_RADIUS_M = 100                                                   # generated all the vehicles data but highlighted thoese which are close to the intersection

# Save only records with speed <= this value
MAX_SPEED_MPS = 14.0

# Optional filtering:
# False -> save all active vehicles in the network
# True  -> save only vehicles within NEAR_JUNCTION_RADIUS_M of the target junction
SAVE_ONLY_NEAR_JUNCTION = False                                                 # selected false mean save all the data in the map           

# Print one progress line every N simulation steps
PRINT_EVERY_N_STEPS = 50


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def infer_vehicle_group(vehicle_id: str) -> str:
    """Simple label based on your current trip naming convention."""
    return "targeted" if str(vehicle_id).startswith("targeted_") else "background"


def safe_get_vehicle_type(vehicle_id: str) -> str:
    try:
        return traci.vehicle.getTypeID(vehicle_id)
    except Exception:
        return "unknown"


def main():
    if not SUMO_CONFIG.exists():
        print(f"ERROR: SUMO config file not found:\n{SUMO_CONFIG}")
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    sumo_cmd = [
        SUMO_BINARY,
        "-c",
        str(SUMO_CONFIG),
        "--start",
        "--quit-on-end",
    ]

    try:
        print("Starting SUMO...")
        traci.start(sumo_cmd)
        print("Connected to SUMO.\n")

        junction_ids = set(traci.junction.getIDList())
        if INTERSECTION_ID not in junction_ids:
            print(f"ERROR: Junction '{INTERSECTION_ID}' not found in this network.")
            traci.close()
            sys.exit(1)

        # Target junction/intersection coordinates
        jx, jy = traci.junction.getPosition(INTERSECTION_ID)

        print(
            f"Tracking junction {INTERSECTION_ID} at x={jx:.2f}, y={jy:.2f} | "
            f"near-radius={NEAR_JUNCTION_RADIUS_M:.1f} m | "
            f"max-speed-filter={MAX_SPEED_MPS:.1f} m/s\n"
        )

        total_rows = 0
        skipped_speed_rows = 0
        step = 0

        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "time",
                "vehicle_id",
                "vehicle_group",
                "vehicle_type",
                "x",
                "y",
                "speed_mps",
                "acceleration_mps2",
                "angle_deg",
                "edge_id",
                "lane_id",
                "lane_position_m",
                "target_junction_x",
                "target_junction_y",
                "distance_to_junction_center_m",
                "is_near_target_junction"
            ])

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()

                sim_time = traci.simulation.getTime()
                vehicle_ids = traci.vehicle.getIDList()
                saved_this_step = 0
                skipped_speed_this_step = 0

                for vid in vehicle_ids:
                    speed = traci.vehicle.getSpeed(vid)

                    # Initial speed filtering: do not save records above 14 m/s
                    if speed > MAX_SPEED_MPS:
                        skipped_speed_rows += 1
                        skipped_speed_this_step += 1
                        continue

                    x, y = traci.vehicle.getPosition(vid)
                    acc = traci.vehicle.getAcceleration(vid)
                    angle = traci.vehicle.getAngle(vid)
                    edge_id = traci.vehicle.getRoadID(vid)
                    lane_id = traci.vehicle.getLaneID(vid)
                    lane_pos = traci.vehicle.getLanePosition(vid)

                    dist = euclidean_distance(x, y, jx, jy)
                    is_near = 1 if dist <= NEAR_JUNCTION_RADIUS_M else 0

                    if SAVE_ONLY_NEAR_JUNCTION and not is_near:
                        continue

                    writer.writerow([
                        round(sim_time, 2),
                        vid,
                        infer_vehicle_group(vid),
                        safe_get_vehicle_type(vid),
                        round(x, 2),
                        round(y, 2),
                        round(speed, 2),
                        round(acc, 2),
                        round(angle, 2),
                        edge_id,
                        lane_id,
                        round(lane_pos, 2),
                        round(jx, 2),
                        round(jy, 2),
                        round(dist, 2),
                        is_near,
                    ])

                    saved_this_step += 1
                    total_rows += 1

                if step % PRINT_EVERY_N_STEPS == 0:
                    print(
                        f"step={step} | time={sim_time:.1f}s | "
                        f"active={len(vehicle_ids)} | "
                        f"saved={saved_this_step} | "
                        f"skipped_speed={skipped_speed_this_step} | "
                        f"total_rows={total_rows}"
                    )

                step += 1

        traci.close()

        print(f"\nDataset saved successfully:\n{OUTPUT_CSV}")
        print(f"Total rows written: {total_rows}")
        print(f"Rows skipped because speed_mps > {MAX_SPEED_MPS}: {skipped_speed_rows}")


    except Exception as e:
        print("\nERROR while running SUMO:")
        print(e)
        try:
            traci.close()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
