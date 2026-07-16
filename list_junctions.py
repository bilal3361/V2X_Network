import sys
from pathlib import Path

import traci

SUMO_BINARY = "sumo-gui"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_CONFIG = PROJECT_ROOT / "osm.sumocfg"


def main():
    if not SUMO_CONFIG.exists():
        print(f"ERROR: SUMO config file not found:\n{SUMO_CONFIG}")
        sys.exit(1)

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

        junction_ids = traci.junction.getIDList()
        print(f"Total junctions found: {len(junction_ids)}\n")

        for jid in junction_ids:
            x, y = traci.junction.getPosition(jid)
            print(f"junction_id={jid} | x={x:.2f} | y={y:.2f}")

        traci.close()
        print("\nDone.")

    except Exception as e:
        print("\nERROR:")
        print(e)
        try:
            traci.close()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
