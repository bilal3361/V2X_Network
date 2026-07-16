import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "targeted_720.trips.xml"

INCOMING_EDGES = [
    "1274037341#1",
    "179750964#5",
    "409634167#1",
]

OUTGOING_EDGES = [
    "179750980#0",
    "409634167#3",
    "858712304#0",
]

TOTAL_TARGETED = 720
SIM_START = 0
SIM_END = 3600


def main():
    random.seed(42)
    depart_step = (SIM_END - SIM_START) / TOTAL_TARGETED

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write("<routes>\n")
        f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="20" guiShape="passenger"/>\n\n')

        for i in range(TOTAL_TARGETED):
            from_edge = random.choice(INCOMING_EDGES)
            to_edge = random.choice(OUTGOING_EDGES)
            depart = round(SIM_START + i * depart_step, 2)

            f.write(
                f'    <trip id="targeted_{i}" type="car" depart="{depart}" '
                f'from="{from_edge}" to="{to_edge}"/>\n'
            )

        f.write("</routes>\n")

    print(f"Created targeted trips file:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
