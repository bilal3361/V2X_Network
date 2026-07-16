from __future__ import annotations

import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = PROJECT_ROOT / "scenario1_50.routes.xml"
CONFLICT_GROUP_PATH = PROJECT_ROOT / "data" / "scenario1_conflict_groups.csv"

VEHICLE_COUNT = 50
CONFLICT_GROUP_COUNT = 15
RANDOM_SEED = 8408
VEHICLE_SPEED_MPS = 13.89
TARGET_JUNCTION_ID = "cluster_255722000_4115305935"

VTYPE = {
    "id": "scenario1_collision_car",
    "vClass": "passenger",
    "guiShape": "passenger",
    "length": "5.00",
    "width": "1.80",
    "minGap": "0",
    "maxSpeed": f"{VEHICLE_SPEED_MPS:.2f}",
    "speedFactor": "1.0",
    "speedDev": "0",
    "accel": "6.0",
    "decel": "9.0",
    "emergencyDecel": "12.0",
    "apparentDecel": "12.0",
    "tau": "0.1",
    "sigma": "0",
    "lcStrategic": "0",
    "lcCooperative": "0",
    "lcSpeedGain": "0",
    "lcKeepRight": "0",
    "lcAssertive": "10",
    "jmIgnoreFoeProb": "1",
    "jmIgnoreJunctionFoeProb": "1",
    "jmIgnoreFoeSpeed": "100",
    "jmIgnoreKeepClearTime": "0",
    "jmStoplineGap": "0",
    "collisionMinGapFactor": "0",
}

MOVEMENTS = {
    "east_to_west": {
        "edges": "179750964#5 858712304#0",
        "depart_lane": "1",
        "approach_time_s": 8.7,
        "color": "red",
    },
    "east_to_south": {
        "edges": "179750964#5 179750980#0",
        "depart_lane": "0",
        "approach_time_s": 8.0,
        "color": "orange",
    },
    "south_to_east": {
        "edges": "1274037341#1 409634167#3",
        "depart_lane": "0",
        "approach_time_s": 5.3,
        "color": "cyan",
    },
    "south_to_north": {
        "edges": "1274037341#1 179750980#0",
        "depart_lane": "0",
        "approach_time_s": 6.5,
        "color": "blue",
    },
    "south_to_west": {
        "edges": "1274037341#1 858712304#0",
        "depart_lane": "0",
        "approach_time_s": 5.3,
        "color": "magenta",
    },
    "west_to_east": {
        "edges": "409634167#1 409634167#3",
        "depart_lane": "0",
        "approach_time_s": 3.4,
        "color": "yellow",
    },
    "west_to_south": {
        "edges": "409634167#1 179750980#0",
        "depart_lane": "1",
        "approach_time_s": 3.4,
        "color": "green",
    },
}


def add_vehicle(
    vehicles: list[dict[str, object]],
    movement_name: str,
    conflict_time_s: float,
    group_id: str,
    purpose: str,
    depart_jitter_s: float = 0.0,
) -> None:
    movement = MOVEMENTS[movement_name]
    depart = max(0.0, conflict_time_s - float(movement["approach_time_s"]) + depart_jitter_s)
    vehicles.append(
        {
            "type": VTYPE["id"],
            "depart": round(depart, 1),
            "departLane": movement["depart_lane"],
            "departPos": "0",
            "departSpeed": f"{VEHICLE_SPEED_MPS:.2f}",
            "color": movement["color"],
            "edges": movement["edges"],
            "movement": movement_name,
            "expected_conflict_time_s": round(conflict_time_s, 1),
            "group_id": group_id,
            "purpose": purpose,
        }
    )


def is_depart_slot_available(
    vehicles: list[dict[str, object]],
    movement_name: str,
    depart_s: float,
    min_gap_s: float = 3.0,
) -> bool:
    for vehicle in vehicles:
        if vehicle["movement"] != movement_name:
            continue
        if abs(float(vehicle["depart"]) - depart_s) < min_gap_s:
            return False
    return True


def make_random_conflict_times(rng: random.Random) -> list[float]:
    start_s = rng.uniform(42.0, 48.0)
    gap_s = 14.0
    times = [
        round(start_s + index * gap_s + rng.uniform(-1.5, 1.5), 1)
        for index in range(CONFLICT_GROUP_COUNT)
    ]
    return sorted(times)


def estimated_arrival_time(vehicle: dict[str, object]) -> float:
    expected = vehicle.get("expected_conflict_time_s", "")
    if expected != "":
        return float(expected)
    return float(vehicle["depart"]) + float(MOVEMENTS[str(vehicle["movement"])]["approach_time_s"])


def is_background_arrival_clear(
    movement_name: str,
    depart_s: float,
    conflict_times: list[float],
    vehicles: list[dict[str, object]],
    conflict_buffer_s: float = 6.0,
    traffic_buffer_s: float = 4.0,
) -> bool:
    arrival_s = depart_s + float(MOVEMENTS[movement_name]["approach_time_s"])
    if any(abs(arrival_s - conflict_time) <= conflict_buffer_s for conflict_time in conflict_times):
        return False
    return all(abs(arrival_s - estimated_arrival_time(vehicle)) > traffic_buffer_s for vehicle in vehicles)


def make_background_arrival_times(
    rng: random.Random,
    conflict_times: list[float],
    count: int,
) -> list[float]:
    candidates = [
        float(slot)
        for slot in range(12, 286, 5)
        if all(abs(float(slot) - conflict_time) > 4.0 for conflict_time in conflict_times)
    ]
    if len(candidates) < count:
        raise RuntimeError("Not enough safe random background slots are available.")
    selected = sorted(rng.sample(candidates, count))
    return [round(value + rng.uniform(-0.8, 0.8), 1) for value in selected]


def build_vehicles() -> list[dict[str, object]]:
    rng = random.Random(RANDOM_SEED)
    vehicles: list[dict[str, object]] = []

    # Random-looking conflict pairs: the pair timing repeats the proven
    # Scenario4 collision pattern, but the pair times are jittered across the
    # full run.
    conflict_times = make_random_conflict_times(rng)
    for group_index, conflict_time in enumerate(conflict_times, start=1):
        group_id = f"scenario4_style_pair_{group_index:02d}"
        for movement_name in ["east_to_west", "south_to_north"]:
            add_vehicle(
                vehicles,
                movement_name,
                conflict_time,
                group_id,
                "intentional_conflict",
                depart_jitter_s=rng.uniform(-0.15, 0.15),
            )

    background_movements = list(MOVEMENTS)
    background_count = VEHICLE_COUNT - len(vehicles)
    for arrival_time in make_background_arrival_times(rng, conflict_times, background_count):
        movement_name = rng.choice(background_movements)
        for _attempt in range(20):
            movement = MOVEMENTS[movement_name]
            depart = round(max(0.0, arrival_time - float(movement["approach_time_s"])), 1)
            if is_depart_slot_available(vehicles, movement_name, depart):
                break
            movement_name = rng.choice(background_movements)
        else:
            raise RuntimeError("Could not assign a background vehicle movement with enough depart spacing.")
        movement = MOVEMENTS[movement_name]
        vehicles.append(
            {
                "type": VTYPE["id"],
                "depart": depart,
                "departLane": movement["depart_lane"],
                "departPos": "0",
                "departSpeed": f"{VEHICLE_SPEED_MPS:.2f}",
                "color": movement["color"],
                "edges": movement["edges"],
                "movement": movement_name,
                "expected_conflict_time_s": "",
                "group_id": "random_background",
                "purpose": "random_background",
            }
        )

    if len(vehicles) != VEHICLE_COUNT:
        raise RuntimeError(f"Expected {VEHICLE_COUNT} vehicles, generated {len(vehicles)}")

    vehicles = sorted(vehicles, key=lambda item: (float(item["depart"]), str(item["movement"])))
    for index, vehicle in enumerate(vehicles, start=1):
        vehicle["id"] = f"targeted_vehicle_{index:03d}"
    return vehicles


def write_routes(vehicles: list[dict[str, object]]) -> None:
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    ET.SubElement(routes, "vType", VTYPE)

    for vehicle in vehicles:
        vehicle_attrs = {
            key: str(vehicle[key])
            for key in ["id", "type", "depart", "departLane", "departPos", "departSpeed", "color"]
        }
        vehicle_el = ET.SubElement(routes, "vehicle", vehicle_attrs)
        ET.SubElement(vehicle_el, "route", {"edges": vehicle["edges"]})

    tree = ET.ElementTree(routes)
    ET.indent(tree, space="    ")
    tree.write(ROUTE_PATH, encoding="UTF-8", xml_declaration=True)


def write_conflict_groups(vehicles: list[dict[str, object]]) -> None:
    CONFLICT_GROUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group_id",
        "vehicle_id",
        "movement",
        "route_edges",
        "depart",
        "expected_conflict_time_s",
        "target_junction_id",
        "purpose",
        "note",
    ]
    with CONFLICT_GROUP_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for vehicle in vehicles:
            writer.writerow(
                {
                    "group_id": vehicle["group_id"],
                    "vehicle_id": vehicle["id"],
                    "movement": vehicle["movement"],
                    "route_edges": vehicle["edges"],
                    "depart": vehicle["depart"],
                    "expected_conflict_time_s": vehicle["expected_conflict_time_s"],
                    "target_junction_id": TARGET_JUNCTION_ID,
                    "purpose": vehicle["purpose"],
                    "note": (
                        "Random traffic vehicle."
                        if vehicle["purpose"] == "random_background"
                        else "Randomly placed intentional arrival-time conflict for warning/protection demonstration."
                    ),
                }
            )


def main() -> int:
    vehicles = build_vehicles()
    write_routes(vehicles)
    write_conflict_groups(vehicles)

    print(f"Wrote {ROUTE_PATH}")
    print(f"Wrote {CONFLICT_GROUP_PATH}")
    print(f"vehicle_count={len(vehicles)}")
    print(f"random_seed={RANDOM_SEED}")
    print("vehicle_ids=targeted_vehicle_001..targeted_vehicle_050")
    print("movement_distribution:")
    for movement_name in MOVEMENTS:
        count = sum(1 for vehicle in vehicles if vehicle["movement"] == movement_name)
        print(f"  {movement_name}: {count}")
    print("purpose_distribution:")
    for purpose in ["intentional_conflict", "random_background"]:
        count = sum(1 for vehicle in vehicles if vehicle["purpose"] == purpose)
        print(f"  {purpose}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
