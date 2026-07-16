import sys
from pathlib import Path

import traci

SUMO_BINARY = "sumo-gui"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_CONFIG = PROJECT_ROOT / "osm.sumocfg"
INTERSECTION_ID = "cluster_255722000_4115305935"


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

        junction_ids = set(traci.junction.getIDList())
        if INTERSECTION_ID not in junction_ids:
            print(f"ERROR: Intersection '{INTERSECTION_ID}' not found.")
            traci.close()
            sys.exit(1)

        print(f"Intersection: {INTERSECTION_ID}\n")

        incoming = traci.junction.getIncomingEdges(INTERSECTION_ID)
        outgoing = traci.junction.getOutgoingEdges(INTERSECTION_ID)

        print("Incoming edges:")
        for edge in incoming:
            print(edge)

        print("\nOutgoing edges:")
        for edge in outgoing:
            print(edge)

        traci.close()
        print("\nDone.")

    except Exception as e:
        print("ERROR:")
        print(e)
        try:
            traci.close()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
