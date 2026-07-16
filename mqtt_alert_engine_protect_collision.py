from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any

import mqtt_alert_engine as base_engine
from v2x_task5_common import (
    PROJECT_ROOT,
    TrajectoryPredictor,
    VehicleHistoryStore,
    build_pair_alerts,
    risk_topic_suffix,
    utc_now_iso,
)


DEFAULT_PROTECTED_SUMO_CONFIG = PROJECT_ROOT / "osm.sumocfg"
PROTECTION_LOG_PATH = PROJECT_ROOT / "data" / "scenario1_mqtt_protection_log.csv"
COLLISION_DIAGNOSIS_LOG_PATH = PROJECT_ROOT / "data" / "scenario1_collision_diagnosis_log.csv"

TARGET_JUNCTION_ID = ""
LANE_LEADS_TO_TARGET_CACHE: dict[str, bool] = {}

PREPARE_SPEED_MPS = 2.0
WAIT_SPEED_MPS = 0.3
SLOWDOWN_DURATION_S = 2.0
SPEED_COMMAND_REFRESH_S = 0.2
SAFE_TIME_GAP_S = 2.0

APPROACH_ZONE_RADIUS_M = 150.0
STOP_ZONE_RADIUS_M = 18.0
CONFLICT_ZONE_RADIUS_M = 20.0
CLEAR_ZONE_RADIUS_M = 25.0
PREPARE_TO_WAIT_DISTANCE_M = 85.0
WAIT_AT_ENTRY_DISTANCE_M = 25.0
CONFLICT_ENTRY_GUARD_DISTANCE_M = 9.0
ENTRY_HOLD_BUFFER_M = 3.5
ENTRY_HOLD_DISTANCE_M = CONFLICT_ENTRY_GUARD_DISTANCE_M + ENTRY_HOLD_BUFFER_M
ENTRY_CREEP_SPEED_MPS = 2.0
COMFORTABLE_DECEL_MPS2 = 5.0
MAX_CONTROL_DECEL_MPS2 = 12.0
HOLD_SPEED_THRESHOLD_MPS = 0.7
MIN_SLOWDOWN_DURATION_S = 0.4
MAX_SLOWDOWN_DURATION_S = 2.5

PASSED_RELEASE_MARGIN_M = 5.0
HOLD_LOG_INTERVAL_S = 1.0
MAX_WAIT_S = 8.0
COOLDOWN_S = 5.0
PROTECTED_SPEED_MODE = 31
PROTECTED_MIN_GAP_M = 2.5
PROTECTED_TAU_S = 1.0

LOG_FIELDS = [
    "simulation_time",
    "risk_level",
    "vehicle_1",
    "vehicle_2",
    "action",
    "priority_vehicle_id",
    "yielding_vehicle_id",
    "reservation_queue",
    "conflict_group_vehicle_ids",
    "predicted_arrival_time_s",
    "conflict_zone_vehicle_ids",
    "target_speed_mps",
    "release_reason",
    "note",
]

DIAGNOSIS_LOG_FIELDS = [
    "simulation_time",
    "vehicle_1",
    "vehicle_2",
    "lane",
    "junction_or_edge",
    "inside_target_junction",
    "alert_seen_before_collision",
    "risk_level_before_collision",
    "protection_applied_vehicle_1",
    "protection_applied_vehicle_2",
    "reservation_queue_before_collision",
    "conflict_zone_vehicle_ids_before_collision",
    "suspected_cause",
]


def parse_args() -> Any:
    base_engine.DEFAULT_SUMO_CONFIG = DEFAULT_PROTECTED_SUMO_CONFIG
    args = base_engine.parse_args()
    if args.client_id == "v2x-task5-alert-engine":
        args.client_id = "v2x-scenario1-protected-mqtt"
    return args


def open_protection_log(path: Path = PROTECTION_LOG_PATH) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
    writer.writeheader()
    handle.flush()
    return handle, writer


def open_collision_diagnosis_log(
    path: Path = COLLISION_DIAGNOSIS_LOG_PATH,
) -> tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=DIAGNOSIS_LOG_FIELDS)
    writer.writeheader()
    handle.flush()
    return handle, writer


def wait_for_mqtt_connection(client: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        try:
            if client.is_connected():
                return
        except Exception:
            pass
        time.sleep(0.05)
    raise RuntimeError("MQTT client did not connect before the protected engine started publishing alerts.")


def add_mqtt_comparison_fields(payload: dict[str, Any]) -> dict[str, Any]:
    if hasattr(base_engine, "add_mqtt_comparison_fields"):
        return base_engine.add_mqtt_comparison_fields(payload)
    payload["protocol"] = "mqtt"
    payload["sent_perf_time"] = time.perf_counter()
    payload["sent_wall_time_utc"] = utc_now_iso()
    if hasattr(base_engine, "format_cest_time"):
        payload["sent_wall_time_cest"] = base_engine.format_cest_time(payload["sent_wall_time_utc"])
    return payload


def format_optional_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def write_protection_log(
    writer: csv.DictWriter,
    handle: Any,
    sim_time: float,
    action: str,
    priority_vehicle_id: str = "",
    yielding_vehicle_id: str = "",
    reservation_queue: list[str] | None = None,
    conflict_group_vehicle_ids: list[str] | None = None,
    predicted_arrival_time_s: float | None = None,
    conflict_zone_vehicle_ids: list[str] | None = None,
    target_speed_mps: float | None = None,
    release_reason: str = "",
    note: str = "",
    alert: dict[str, Any] | None = None,
) -> None:
    zone_ids = sorted(map(str, conflict_zone_vehicle_ids or []))
    queue_ids = list(map(str, reservation_queue or []))
    group_ids = sorted(map(str, conflict_group_vehicle_ids or []))
    alert_payload = alert or {}
    writer.writerow(
        {
            "simulation_time": f"{float(sim_time):.2f}",
            "risk_level": str(alert_payload.get("risk_level", "")),
            "vehicle_1": str(alert_payload.get("vehicle_1", "")),
            "vehicle_2": str(alert_payload.get("vehicle_2", "")),
            "action": action,
            "priority_vehicle_id": priority_vehicle_id,
            "yielding_vehicle_id": yielding_vehicle_id,
            "reservation_queue": ";".join(queue_ids),
            "conflict_group_vehicle_ids": ";".join(group_ids),
            "predicted_arrival_time_s": format_optional_float(predicted_arrival_time_s),
            "conflict_zone_vehicle_ids": ";".join(zone_ids),
            "target_speed_mps": format_optional_float(target_speed_mps),
            "release_reason": release_reason,
            "note": note,
        }
    )
    handle.flush()


def get_distance_to_junction(
    traci: Any,
    vehicle_id: str,
    junction_x: float,
    junction_y: float,
) -> float | None:
    try:
        x, y = traci.vehicle.getPosition(vehicle_id)
    except Exception:
        return None
    dx = float(x) - float(junction_x)
    dy = float(y) - float(junction_y)
    return ((dx * dx) + (dy * dy)) ** 0.5


def lane_is_target_internal(lane_id: str) -> bool:
    return bool(TARGET_JUNCTION_ID) and str(lane_id).startswith(f":{TARGET_JUNCTION_ID}_")


def is_on_target_internal_lane(traci: Any, vehicle_id: str) -> bool:
    try:
        road_id = str(traci.vehicle.getRoadID(vehicle_id))
        lane_id = str(traci.vehicle.getLaneID(vehicle_id))
    except Exception:
        return False
    return lane_is_target_internal(road_id) or lane_is_target_internal(lane_id)


def lane_leads_to_target_junction(traci: Any, lane_id: str) -> bool:
    lane_id = str(lane_id)
    cached = LANE_LEADS_TO_TARGET_CACHE.get(lane_id)
    if cached is not None:
        return cached

    try:
        links = traci.lane.getLinks(lane_id, extended=True)
    except TypeError:
        try:
            links = traci.lane.getLinks(lane_id)
        except Exception:
            links = []
    except Exception:
        links = []

    target_prefix = f":{TARGET_JUNCTION_ID}_"
    for link in links:
        for value in link:
            if isinstance(value, str) and value.startswith(target_prefix):
                LANE_LEADS_TO_TARGET_CACHE[lane_id] = True
                return True

    LANE_LEADS_TO_TARGET_CACHE[lane_id] = False
    return False


def distance_to_junction_entry(traci: Any, vehicle_id: str) -> float | None:
    if is_on_target_internal_lane(traci, vehicle_id):
        return 0.0
    try:
        lane_id = str(traci.vehicle.getLaneID(vehicle_id))
    except Exception:
        return None
    if not lane_leads_to_target_junction(traci, lane_id):
        return None
    try:
        lane_length_m = float(traci.lane.getLength(lane_id))
        lane_position_m = float(traci.vehicle.getLanePosition(vehicle_id))
    except Exception:
        return None
    return max(0.0, lane_length_m - lane_position_m)


def get_vehicles_in_conflict_zone(
    traci: Any,
    junction_x: float,
    junction_y: float,
    radius_m: float = CONFLICT_ZONE_RADIUS_M,
) -> list[str]:
    zone_vehicle_ids: list[str] = []
    for vehicle_id in map(str, traci.vehicle.getIDList()):
        entry_distance_m = distance_to_junction_entry(traci, vehicle_id)
        if is_on_target_internal_lane(traci, vehicle_id) or (
            entry_distance_m is not None and entry_distance_m <= CONFLICT_ENTRY_GUARD_DISTANCE_M
        ):
            zone_vehicle_ids.append(vehicle_id)
    return sorted(zone_vehicle_ids)


def is_vehicle_inside_conflict_zone(
    traci: Any,
    vehicle_id: str,
    junction_x: float,
    junction_y: float,
) -> bool:
    entry_distance_m = distance_to_junction_entry(traci, vehicle_id)
    return is_on_target_internal_lane(traci, vehicle_id) or (
        entry_distance_m is not None and entry_distance_m <= CONFLICT_ENTRY_GUARD_DISTANCE_M
    )


def update_vehicle_progress_state(state: dict[str, Any], distance_m: float | None) -> None:
    if distance_m is None:
        return
    previous_min = state.get("min_distance_m")
    state["min_distance_m"] = float(distance_m) if previous_min is None else min(float(previous_min), float(distance_m))
    if distance_m <= STOP_ZONE_RADIUS_M:
        state["entered_stop_zone"] = True
    if distance_m <= CONFLICT_ZONE_RADIUS_M:
        state["entered_conflict_zone"] = True
    if distance_m <= CLEAR_ZONE_RADIUS_M:
        state["entered_clear_zone"] = True


def vehicle_is_before_or_inside_target_junction(traci: Any, vehicle_id: str) -> bool:
    return is_on_target_internal_lane(traci, vehicle_id) or distance_to_junction_entry(traci, vehicle_id) is not None


def is_target_internal_lane(lane_id: str) -> bool:
    return lane_is_target_internal(str(lane_id))


def lane_junction_or_edge(lane_id: str) -> str:
    lane_id = str(lane_id)
    if lane_id.startswith(":"):
        return lane_id.split("_", 1)[0].lstrip(":")
    return lane_id.rsplit("_", 1)[0]


def risk_priority(risk_level: str) -> int:
    return 2 if str(risk_level).upper() == "HIGH" else 1 if str(risk_level).upper() == "LOW" else 0


def normalize_pair(vehicle_1: str, vehicle_2: str) -> tuple[str, str]:
    pair = sorted([str(vehicle_1), str(vehicle_2)])
    return pair[0], pair[1]


def remember_speed_mode(traci: Any, vehicle_id: str, state: dict[str, Any]) -> None:
    if state.get("original_speed_mode") is None:
        state["original_speed_mode"] = int(traci.vehicle.getSpeedMode(vehicle_id))


def apply_manual_speed_control(
    traci: Any,
    vehicle_id: str,
    state: dict[str, Any],
    target_speed_mps: float,
    sim_time: float,
    force: bool = False,
    duration_s: float | None = None,
) -> bool:
    previous_target = state.get("target_speed_mps")
    previous_command_time = state.get("last_speed_command_time_s")
    if (
        not force
        and previous_target is not None
        and abs(float(previous_target) - float(target_speed_mps)) < 0.05
        and previous_command_time is not None
        and (float(sim_time) - float(previous_command_time)) < SPEED_COMMAND_REFRESH_S
    ):
        return False

    remember_speed_mode(traci, vehicle_id, state)
    traci.vehicle.setSpeedMode(vehicle_id, PROTECTED_SPEED_MODE)
    traci.vehicle.slowDown(
        vehicle_id,
        float(target_speed_mps),
        SLOWDOWN_DURATION_S if duration_s is None else float(duration_s),
    )
    state["target_speed_mps"] = float(target_speed_mps)
    state["last_speed_command_time_s"] = float(sim_time)
    return True


def release_manual_speed_control(traci: Any, vehicle_id: str, state: dict[str, Any]) -> None:
    traci.vehicle.setSpeed(vehicle_id, -1)
    original_speed_mode = state.get("original_speed_mode")
    if original_speed_mode is not None:
        traci.vehicle.setSpeedMode(vehicle_id, int(original_speed_mode))


def apply_entry_wait_control(
    traci: Any,
    vehicle_id: str,
    state: dict[str, Any],
    sim_time: float,
) -> None:
    remember_speed_mode(traci, vehicle_id, state)
    traci.vehicle.setSpeedMode(vehicle_id, 0)
    traci.vehicle.slowDown(vehicle_id, 0.0, 0.1)
    traci.vehicle.setSpeed(vehicle_id, 0.0)
    state["target_speed_mps"] = 0.0
    state["last_speed_command_time_s"] = float(sim_time)


def clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(float(min_value), min(float(max_value), float(value)))


def vehicle_decel_limits(traci: Any, vehicle_id: str) -> tuple[float, float]:
    try:
        vehicle_decel = float(traci.vehicle.getDecel(vehicle_id))
    except Exception:
        try:
            vehicle_type_id = traci.vehicle.getTypeID(vehicle_id)
            vehicle_decel = float(traci.vehicletype.getDecel(vehicle_type_id))
        except Exception:
            vehicle_decel = MAX_CONTROL_DECEL_MPS2

    max_control_decel = clamp_float(vehicle_decel, COMFORTABLE_DECEL_MPS2, MAX_CONTROL_DECEL_MPS2)
    comfortable_decel = min(COMFORTABLE_DECEL_MPS2, max_control_decel * 0.75)
    return comfortable_decel, max_control_decel


def apply_smooth_entry_yield_control(
    traci: Any,
    vehicle_id: str,
    state: dict[str, Any],
    sim_time: float,
    entry_distance_m: float,
) -> tuple[float | None, str]:
    try:
        current_speed_mps = max(0.0, float(traci.vehicle.getSpeed(vehicle_id)))
    except Exception:
        current_speed_mps = 0.0

    remaining_to_hold_m = max(0.0, float(entry_distance_m) - ENTRY_HOLD_DISTANCE_M)
    comfortable_decel, max_control_decel = vehicle_decel_limits(traci, vehicle_id)

    if remaining_to_hold_m <= 0.05 and current_speed_mps <= HOLD_SPEED_THRESHOLD_MPS:
        apply_entry_wait_control(traci, vehicle_id, state, sim_time)
        return 0.0, (
            f"wait_near_junction_entry entry_distance_m={entry_distance_m:.1f} "
            f"hold_distance_m={ENTRY_HOLD_DISTANCE_M:.1f} speed_mps={current_speed_mps:.2f}"
        )

    if remaining_to_hold_m <= 0.05:
        target_speed_mps = WAIT_SPEED_MPS
        decel = max_control_decel
        duration_s = clamp_float(
            (current_speed_mps - target_speed_mps) / max(decel, 0.1),
            MIN_SLOWDOWN_DURATION_S,
            MAX_SLOWDOWN_DURATION_S,
        )
        apply_manual_speed_control(
            traci,
            vehicle_id,
            state,
            target_speed_mps,
            sim_time,
            force=True,
            duration_s=duration_s,
        )
        return target_speed_mps, (
            f"final_brake_before_guard entry_distance_m={entry_distance_m:.1f} "
            f"hold_distance_m={ENTRY_HOLD_DISTANCE_M:.1f} speed_mps={current_speed_mps:.2f} "
            f"target_speed_mps={target_speed_mps:.2f} duration_s={duration_s:.2f}"
        )

    target_speed_mps = (2.0 * comfortable_decel * remaining_to_hold_m) ** 0.5
    target_speed_mps = clamp_float(target_speed_mps, WAIT_SPEED_MPS, PREPARE_SPEED_MPS)

    if current_speed_mps <= target_speed_mps + 0.2:
        return None, (
            f"smooth_profile_no_brake entry_distance_m={entry_distance_m:.1f} "
            f"remaining_to_hold_m={remaining_to_hold_m:.1f} speed_mps={current_speed_mps:.2f} "
            f"profile_speed_mps={target_speed_mps:.2f}"
        )

    required_decel = max(
        0.0,
        ((current_speed_mps * current_speed_mps) - (target_speed_mps * target_speed_mps))
        / max(2.0 * remaining_to_hold_m, 0.1),
    )
    control_decel = clamp_float(required_decel, comfortable_decel, max_control_decel)
    duration_s = clamp_float(
        (current_speed_mps - target_speed_mps) / max(control_decel, 0.1),
        MIN_SLOWDOWN_DURATION_S,
        MAX_SLOWDOWN_DURATION_S,
    )
    apply_manual_speed_control(
        traci,
        vehicle_id,
        state,
        target_speed_mps,
        sim_time,
        duration_s=duration_s,
    )
    return target_speed_mps, (
        f"smooth_deceleration_to_entry entry_distance_m={entry_distance_m:.1f} "
        f"remaining_to_hold_m={remaining_to_hold_m:.1f} speed_mps={current_speed_mps:.2f} "
        f"target_speed_mps={target_speed_mps:.2f} decel_mps2={control_decel:.2f} "
        f"duration_s={duration_s:.2f}"
    )


def configure_protected_vehicle_types(traci: Any) -> None:
    for vehicle_type_id in map(str, traci.vehicletype.getIDList()):
        try:
            traci.vehicletype.setMinGap(vehicle_type_id, PROTECTED_MIN_GAP_M)
            traci.vehicletype.setTau(vehicle_type_id, PROTECTED_TAU_S)
            traci.vehicletype.setParameter(vehicle_type_id, "jmIgnoreFoeProb", "0")
            traci.vehicletype.setParameter(vehicle_type_id, "jmIgnoreJunctionFoeProb", "0")
            traci.vehicletype.setParameter(vehicle_type_id, "jmIgnoreFoeSpeed", "0")
            traci.vehicletype.setParameter(vehicle_type_id, "collisionMinGapFactor", "1")
        except Exception:
            continue


def apply_protected_vehicle_parameters(traci: Any, vehicle_id: str) -> None:
    try:
        traci.vehicle.setParameter(vehicle_id, "jmIgnoreFoeProb", "0")
        traci.vehicle.setParameter(vehicle_id, "jmIgnoreJunctionFoeProb", "0")
        traci.vehicle.setParameter(vehicle_id, "jmIgnoreFoeSpeed", "0")
        traci.vehicle.setParameter(vehicle_id, "collisionMinGapFactor", "1")
        traci.vehicle.setSpeedMode(vehicle_id, PROTECTED_SPEED_MODE)
    except Exception:
        pass


def alert_arrival_time(alert: dict[str, Any], vehicle_id: str) -> float | None:
    if vehicle_id == str(alert.get("vehicle_1", "")):
        value = alert.get("vehicle_1_arrival_time_s")
    elif vehicle_id == str(alert.get("vehicle_2", "")):
        value = alert.get("vehicle_2_arrival_time_s")
    else:
        value = None
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def collect_alert_arrivals(
    alerts: list[dict[str, Any]],
    active_vehicle_ids: set[str],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    arrival_times: dict[str, float] = {}
    alert_by_vehicle: dict[str, dict[str, Any]] = {}
    for alert in alerts:
        for vehicle_key in ("vehicle_1", "vehicle_2"):
            vehicle_id = str(alert.get(vehicle_key, ""))
            if vehicle_id not in active_vehicle_ids:
                continue
            arrival_time_s = alert_arrival_time(alert, vehicle_id)
            if arrival_time_s is None:
                continue
            if vehicle_id not in arrival_times or arrival_time_s < arrival_times[vehicle_id]:
                arrival_times[vehicle_id] = arrival_time_s
                alert_by_vehicle[vehicle_id] = alert
    return arrival_times, alert_by_vehicle


def estimate_live_arrival_time(vehicle_id: str, current_states: dict[str, dict[str, Any]], sim_time: float) -> float:
    state = current_states.get(vehicle_id, {})
    try:
        distance_m = float(state.get("distance_to_junction_center_m", 999999.0))
    except (TypeError, ValueError):
        distance_m = 999999.0
    try:
        speed_mps = max(float(state.get("speed_mps", 0.0)), 0.1)
    except (TypeError, ValueError):
        speed_mps = 0.1
    return float(sim_time) + (distance_m / speed_mps)


def ensure_gate_state(gate_state: dict[str, Any], vehicle_id: str, sim_time: float) -> dict[str, Any]:
    controlled: dict[str, dict[str, Any]] = gate_state.setdefault("controlled", {})
    return controlled.setdefault(
        vehicle_id,
        {
            "original_speed_mode": None,
            "first_control_time_s": float(sim_time),
            "last_action": "",
            "last_log_time_s": -999999.0,
            "target_speed_mps": None,
            "last_speed_command_time_s": -999999.0,
            "stop_active": False,
            "stop_edge_id": "",
            "stop_pos_m": None,
        },
    )


def release_gate_vehicle(
    traci: Any,
    gate_state: dict[str, Any],
    vehicle_id: str,
    sim_time: float,
) -> None:
    state = gate_state.setdefault("controlled", {}).pop(vehicle_id, None)
    if state is not None and state.get("stop_active"):
        try:
            traci.vehicle.resume(vehicle_id)
        except Exception:
            pass
    if state is not None and state.get("original_speed_mode") is not None:
        try:
            release_manual_speed_control(traci, vehicle_id, state)
        except Exception:
            pass
    else:
        try:
            traci.vehicle.setSpeed(vehicle_id, -1)
        except Exception:
            pass
    gate_state.setdefault("release_cooldowns", {})[vehicle_id] = float(sim_time)


def apply_gate_wait(
    traci: Any,
    gate_state: dict[str, Any],
    vehicle_id: str,
    sim_time: float,
    distance_m: float | None = None,
) -> tuple[float | None, str]:
    state = ensure_gate_state(gate_state, vehicle_id, sim_time)
    entry_distance_m = distance_to_junction_entry(traci, vehicle_id)
    if entry_distance_m is not None:
        if entry_distance_m > PREPARE_TO_WAIT_DISTANCE_M:
            if state.get("target_speed_mps") is not None:
                release_manual_speed_control(traci, vehicle_id, state)
            state["target_speed_mps"] = None
            state["last_speed_command_time_s"] = float(sim_time)
            return None, f"normal_approach_until_near_entry entry_distance_m={entry_distance_m:.1f}"
        return apply_smooth_entry_yield_control(traci, vehicle_id, state, sim_time, entry_distance_m)

    if distance_m is not None and float(distance_m) > PREPARE_TO_WAIT_DISTANCE_M:
        if state.get("target_speed_mps") is not None:
            release_manual_speed_control(traci, vehicle_id, state)
        state["target_speed_mps"] = None
        state["last_speed_command_time_s"] = float(sim_time)
        return None, "normal_approach_until_near_stop_line"

    if distance_m is not None:
        return apply_smooth_entry_yield_control(traci, vehicle_id, state, sim_time, float(distance_m))

    return None, "skip_no_distance_to_stop_line"


def should_log_action(state: dict[str, Any], action: str, sim_time: float) -> bool:
    return state.get("last_action") != action or (
        float(sim_time) - float(state.get("last_log_time_s", -999999.0))
    ) >= HOLD_LOG_INTERVAL_S


def maybe_log_reservation_action(
    writer: csv.DictWriter,
    handle: Any,
    sim_time: float,
    state: dict[str, Any] | None,
    action: str,
    priority_vehicle_id: str,
    yielding_vehicle_id: str,
    reservation_queue: list[str],
    conflict_group_vehicle_ids: list[str],
    predicted_arrival_time_s: float | None,
    conflict_zone_vehicle_ids: list[str],
    target_speed_mps: float | None,
    release_reason: str,
    note: str,
    alert: dict[str, Any] | None = None,
) -> bool:
    if state is not None and not should_log_action(state, action, sim_time):
        return False
    write_protection_log(
        writer,
        handle,
        sim_time,
        action,
        priority_vehicle_id=priority_vehicle_id,
        yielding_vehicle_id=yielding_vehicle_id,
        reservation_queue=reservation_queue,
        conflict_group_vehicle_ids=conflict_group_vehicle_ids,
        predicted_arrival_time_s=predicted_arrival_time_s,
        conflict_zone_vehicle_ids=conflict_zone_vehicle_ids,
        target_speed_mps=target_speed_mps,
        release_reason=release_reason,
        note=note,
        alert=alert,
    )
    if state is not None:
        state["last_action"] = action
        state["last_log_time_s"] = float(sim_time)
    return True


def maintain_intersection_gate_controls(
    traci: Any,
    writer: csv.DictWriter,
    handle: Any,
    gate_state: dict[str, Any],
    active_vehicle_ids: set[str],
    current_states: dict[str, dict[str, Any]],
    sim_time: float,
    junction_x: float,
    junction_y: float,
    high_alerts: list[dict[str, Any]],
) -> tuple[int, int]:
    newly_controlled_count = 0
    release_count = 0
    gate_state.setdefault("controlled", {})
    gate_state.setdefault("release_cooldowns", {})
    gate_state.setdefault("priority_progress", {})

    controlled_vehicle_ids = set(gate_state.get("controlled", {}))
    for vehicle_id in active_vehicle_ids:
        if vehicle_id in controlled_vehicle_ids:
            continue
        if vehicle_id.startswith("targeted"):
            apply_protected_vehicle_parameters(traci, vehicle_id)

    alert_arrivals, alert_by_vehicle = collect_alert_arrivals(high_alerts, active_vehicle_ids)
    zone_vehicle_ids = [
        vehicle_id
        for vehicle_id in get_vehicles_in_conflict_zone(traci, junction_x, junction_y)
        if vehicle_id in active_vehicle_ids and vehicle_id.startswith("targeted")
    ]

    candidate_vehicle_ids: list[str] = []
    arrival_times: dict[str, float] = {}
    for vehicle_id, state in current_states.items():
        vehicle_id = str(vehicle_id)
        if vehicle_id not in active_vehicle_ids or not vehicle_id.startswith("targeted"):
            continue
        if not vehicle_is_before_or_inside_target_junction(traci, vehicle_id):
            continue
        try:
            distance_m = float(state.get("distance_to_junction_center_m", 999999.0))
        except (TypeError, ValueError):
            continue
        if distance_m > APPROACH_ZONE_RADIUS_M and vehicle_id not in zone_vehicle_ids:
            continue
        candidate_vehicle_ids.append(vehicle_id)
        arrival_times[vehicle_id] = alert_arrivals.get(
            vehicle_id,
            estimate_live_arrival_time(vehicle_id, current_states, sim_time),
        )

    current_priority_vehicle_id = str(gate_state.get("current_priority_vehicle_id", ""))
    if current_priority_vehicle_id and current_priority_vehicle_id not in active_vehicle_ids:
        current_priority_vehicle_id = ""
        gate_state["current_priority_vehicle_id"] = ""
        gate_state["priority_progress"] = {}

    if current_priority_vehicle_id:
        priority_distance_m = get_distance_to_junction(traci, current_priority_vehicle_id, junction_x, junction_y)
        priority_progress: dict[str, Any] = dict(gate_state.get("priority_progress", {}))
        if not priority_progress:
            priority_progress = {
                "min_distance_m": priority_distance_m,
                "entered_stop_zone": bool(priority_distance_m is not None and priority_distance_m <= STOP_ZONE_RADIUS_M),
                "entered_conflict_zone": bool(priority_distance_m is not None and priority_distance_m <= CONFLICT_ZONE_RADIUS_M),
                "entered_clear_zone": bool(priority_distance_m is not None and priority_distance_m <= CLEAR_ZONE_RADIUS_M),
            }
        update_vehicle_progress_state(priority_progress, priority_distance_m)
        gate_state["priority_progress"] = priority_progress

        priority_has_cleared = (
            priority_distance_m is None
            or not vehicle_is_before_or_inside_target_junction(traci, current_priority_vehicle_id)
            or (
                bool(priority_progress.get("entered_conflict_zone"))
                and not is_vehicle_inside_conflict_zone(traci, current_priority_vehicle_id, junction_x, junction_y)
            )
        )
        if priority_has_cleared:
            release_gate_vehicle(traci, gate_state, current_priority_vehicle_id, sim_time)
            write_protection_log(
                writer,
                handle,
                sim_time,
                "RELEASE_NEXT_VEHICLE",
                priority_vehicle_id=current_priority_vehicle_id,
                yielding_vehicle_id=current_priority_vehicle_id,
                reservation_queue=[current_priority_vehicle_id],
                conflict_group_vehicle_ids=candidate_vehicle_ids,
                predicted_arrival_time_s=arrival_times.get(current_priority_vehicle_id),
                conflict_zone_vehicle_ids=zone_vehicle_ids,
                release_reason="priority_cleared_clear_zone",
                note="Priority vehicle cleared the target conflict/clear zone.",
                alert=alert_by_vehicle.get(current_priority_vehicle_id),
            )
            release_count += 1
            current_priority_vehicle_id = ""
            gate_state["current_priority_vehicle_id"] = ""
            gate_state["priority_progress"] = {}
            gate_state["last_release_time_s"] = float(sim_time)

    sorted_candidates = sorted(
        set(candidate_vehicle_ids),
        key=lambda vehicle_id: (
            arrival_times.get(vehicle_id, 999999.0),
            vehicle_id,
        ),
    )

    if not current_priority_vehicle_id:
        if zone_vehicle_ids:
            current_priority_vehicle_id = zone_vehicle_ids[0]
        elif sorted_candidates:
            current_priority_vehicle_id = sorted_candidates[0]

        if current_priority_vehicle_id:
            gate_state["current_priority_vehicle_id"] = current_priority_vehicle_id
            priority_distance_m = get_distance_to_junction(traci, current_priority_vehicle_id, junction_x, junction_y)
            gate_state["priority_progress"] = {
                "min_distance_m": priority_distance_m,
                "entered_stop_zone": bool(priority_distance_m is not None and priority_distance_m <= STOP_ZONE_RADIUS_M),
                "entered_conflict_zone": bool(priority_distance_m is not None and priority_distance_m <= CONFLICT_ZONE_RADIUS_M),
                "entered_clear_zone": bool(priority_distance_m is not None and priority_distance_m <= CLEAR_ZONE_RADIUS_M),
            }
            gate_state.setdefault("protected_vehicle_ids", set()).add(current_priority_vehicle_id)
            release_gate_vehicle(traci, gate_state, current_priority_vehicle_id, sim_time)
            write_protection_log(
                writer,
                handle,
                sim_time,
                "RELEASE_NEXT_VEHICLE",
                priority_vehicle_id=current_priority_vehicle_id,
                yielding_vehicle_id=current_priority_vehicle_id,
                reservation_queue=sorted_candidates,
                conflict_group_vehicle_ids=sorted_candidates,
                predicted_arrival_time_s=arrival_times.get(current_priority_vehicle_id),
                conflict_zone_vehicle_ids=zone_vehicle_ids,
                release_reason="fcfs_priority_assigned",
                note="Released priority vehicle matching earliest expected arrival time.",
                alert=alert_by_vehicle.get(current_priority_vehicle_id),
            )
            release_count += 1

    reservation_queue = list(sorted_candidates)
    gate_state["reservation_queue"] = reservation_queue
    gate_state["conflict_zone_vehicle_ids"] = zone_vehicle_ids

    for vehicle_id in list(gate_state.get("controlled", {})):
        if vehicle_id not in active_vehicle_ids or vehicle_id not in sorted_candidates:
            release_gate_vehicle(traci, gate_state, vehicle_id, sim_time)

    for vehicle_id in sorted_candidates:
        if vehicle_id == current_priority_vehicle_id:
            continue

        state = ensure_gate_state(gate_state, vehicle_id, sim_time)
        distance_m = get_distance_to_junction(traci, vehicle_id, junction_x, junction_y)
        if distance_m is None:
            continue

        try:
            leader = traci.vehicle.getLeader(vehicle_id, 35.0)
        except Exception:
            leader = None

        if (
            not zone_vehicle_ids
            and vehicle_id not in alert_by_vehicle
            and vehicle_id not in gate_state.get("controlled", {})
        ):
            release_gate_vehicle(traci, gate_state, vehicle_id, sim_time)
            continue

        if (
            leader
            and vehicle_id not in gate_state.get("controlled", {})
            and vehicle_id not in alert_by_vehicle
            and state.get("target_speed_mps") != 0.0
        ):
            if state.get("target_speed_mps") is not None:
                release_manual_speed_control(traci, vehicle_id, state)
                state["target_speed_mps"] = None
            continue

        newly_controlled = state.get("first_logged") is None
        if newly_controlled:
            state["first_logged"] = True
            newly_controlled_count += 1

        if vehicle_id in zone_vehicle_ids:
            action = "SKIP_ALREADY_IN_CONFLICT_ZONE"
            target_speed_mps = None
            release_reason = "already_inside_conflict_zone"
            note = "Vehicle already crossed into boundary; priority cleared to maintain throughput safely."
        else:
            target_speed_mps, stop_note = apply_gate_wait(traci, gate_state, vehicle_id, sim_time, distance_m)
            action = "SLOW_NEAR_ENTRY" if target_speed_mps not in (None, 0.0) else "WAIT_AT_STOP_ZONE"
            release_reason = "conflict_zone_occupied" if zone_vehicle_ids else "reserved_priority_approaching"
            note = f"Vehicle held at stop line for target-junction reservation: {stop_note}."

        maybe_log_reservation_action(
            writer,
            handle,
            sim_time,
            state,
            action,
            current_priority_vehicle_id,
            vehicle_id,
            reservation_queue,
            sorted_candidates,
            arrival_times.get(vehicle_id),
            zone_vehicle_ids,
            target_speed_mps,
            release_reason,
            note,
            alert=alert_by_vehicle.get(vehicle_id),
        )

    return newly_controlled_count, release_count


def update_alert_history(
    alert_history: dict[tuple[str, str], dict[str, Any]],
    alerts: list[dict[str, Any]],
    sim_time: float,
) -> None:
    for alert in alerts:
        vehicle_1 = str(alert.get("vehicle_1", ""))
        vehicle_2 = str(alert.get("vehicle_2", ""))
        if not vehicle_1 or not vehicle_2:
            continue
        pair = normalize_pair(vehicle_1, vehicle_2)
        existing = alert_history.get(pair)
        risk_level = str(alert.get("risk_level", ""))
        if existing is None or risk_priority(risk_level) >= risk_priority(str(existing.get("risk_level", ""))):
            alert_history[pair] = {
                "risk_level": risk_level,
                "simulation_time": float(sim_time),
            }


def collision_suspected_cause(
    alert_seen: bool,
    risk_level: str,
    protection_1: bool,
    protection_2: bool,
    zone_vehicle_ids: list[str],
    inside_target_junction: bool,
) -> str:
    if not inside_target_junction:
        return "outside_target_junction"
    if not alert_seen:
        return "no_alert_before_collision"
    if str(risk_level).upper() == "HIGH" and not (protection_1 or protection_2):
        return "high_alert_without_protection"
    if len(zone_vehicle_ids) > 1:
        return "multiple_vehicles_in_conflict_zone"
    if protection_1 or protection_2:
        return "protection_applied_but_vehicle_entered_before_gate"
    return "unknown"


def write_collision_diagnosis_rows(
    traci: Any,
    writer: csv.DictWriter,
    handle: Any,
    sim_time: float,
    alert_history: dict[tuple[str, str], dict[str, Any]],
    gate_state: dict[str, Any],
    junction_x: float,
    junction_y: float,
) -> int:
    collision_count = 0
    zone_vehicle_ids = get_vehicles_in_conflict_zone(traci, junction_x, junction_y)
    reservation_queue = list(map(str, gate_state.get("reservation_queue", [])))
    protected_vehicle_ids = set(map(str, gate_state.get("protected_vehicle_ids", set())))
    for collision in traci.simulation.getCollisions():
        vehicle_1 = str(collision.collider)
        vehicle_2 = str(collision.victim)
        pair = normalize_pair(vehicle_1, vehicle_2)
        alert_info = alert_history.get(pair, {})
        lane = str(collision.lane)
        inside_target = is_target_internal_lane(lane)
        risk_level = str(alert_info.get("risk_level", ""))
        protection_1 = vehicle_1 in protected_vehicle_ids
        protection_2 = vehicle_2 in protected_vehicle_ids
        writer.writerow(
            {
                "simulation_time": f"{float(sim_time):.2f}",
                "vehicle_1": vehicle_1,
                "vehicle_2": vehicle_2,
                "lane": lane,
                "junction_or_edge": lane_junction_or_edge(lane),
                "inside_target_junction": str(bool(inside_target)),
                "alert_seen_before_collision": str(bool(alert_info)),
                "risk_level_before_collision": risk_level,
                "protection_applied_vehicle_1": str(bool(protection_1)),
                "protection_applied_vehicle_2": str(bool(protection_2)),
                "reservation_queue_before_collision": ";".join(reservation_queue),
                "conflict_zone_vehicle_ids_before_collision": ";".join(zone_vehicle_ids),
                "suspected_cause": collision_suspected_cause(
                    bool(alert_info),
                    risk_level,
                    protection_1,
                    protection_2,
                    zone_vehicle_ids,
                    bool(inside_target),
                ),
            }
        )
        collision_count += 1
    if collision_count:
        handle.flush()
    return collision_count


def build_sumo_command(args: Any) -> list[str]:
    base_cmd = ["--start", "--emergency-insert", "false"]
    if base_engine.is_sumo_gui_binary(args.sumo_binary):
        return [
            args.sumo_binary,
            "-c",
            str(args.sumo_config),
            "--disable-textures",
            "--window-size",
            "1360,820",
            "--window-pos",
            "30,30",
        ] + base_cmd
    return [args.sumo_binary, "-c", str(args.sumo_config), "--quit-on-end"] + base_cmd


def write_sent_alert_if_available(sent_log_writer: Any, sent_log_handle: Any, alert: dict[str, Any]) -> None:
    if sent_log_writer is None or sent_log_handle is None:
        return
    writer = getattr(base_engine, "write_sent_alert_log", None)
    if writer is not None:
        writer(sent_log_writer, sent_log_handle, alert)


def main() -> int:
    global TARGET_JUNCTION_ID, LANE_LEADS_TO_TARGET_CACHE

    args = parse_args()
    TARGET_JUNCTION_ID = str(args.junction_id)
    LANE_LEADS_TO_TARGET_CACHE = {}

    if not args.sumo_config.exists():
        print(f"SUMO config not found: {args.sumo_config}", file=sys.stderr)
        return 1

    protection_handle, protection_writer = open_protection_log()
    diagnosis_handle, diagnosis_writer = open_collision_diagnosis_log()
    sent_log_handle = None
    sent_log_writer = None

    try:
        traci, mqtt = base_engine.import_runtime_dependencies()
        predictor = TrajectoryPredictor(
            model_path=args.model,
            feature_scaler_path=args.feature_scaler,
            target_scaler_path=args.target_scaler,
            metadata_path=args.metadata,
        )
    except Exception as exc:
        protection_handle.close()
        diagnosis_handle.close()
        print(str(exc), file=sys.stderr)
        return 1

    client = base_engine.make_mqtt_client(mqtt, args.client_id)
    client.connect(args.broker_host, args.broker_port, keepalive=60)
    client.loop_start()
    wait_for_mqtt_connection(client)

    if hasattr(base_engine, "open_sent_alert_log"):
        sent_log_handle, sent_log_writer = base_engine.open_sent_alert_log()

    using_sumo_gui = base_engine.is_sumo_gui_binary(args.sumo_binary)
    history_store = VehicleHistoryStore(input_len=predictor.input_len)
    last_publish_times: dict[tuple[str, str, str], float] = {}
    active_episode_states: dict[tuple[str, str], dict[str, Any]] = {}
    alert_history: dict[tuple[str, str], dict[str, Any]] = {}
    gate_state: dict[str, Any] = {
        "current_priority_vehicle_id": "",
        "reservation_queue": [],
        "conflict_group_vehicle_ids": [],
        "conflict_zone_vehicle_ids": [],
        "protected_vehicle_ids": set(),
        "controlled": {},
        "release_cooldowns": {},
        "last_release_time_s": -999999.0,
        "last_logged_signature": None,
    }
    step = 0
    total_alerts = 0
    high_alerts_seen = 0
    protections_applied = 0
    releases = 0
    collision_diagnoses = 0

    try:
        print("Starting protected collision-avoidance engine.")
        print(f"SUMO config: {args.sumo_config}")
        print(f"Protection log: {PROTECTION_LOG_PATH}")
        print(f"Collision diagnosis log: {COLLISION_DIAGNOSIS_LOG_PATH}")
        print("FCFS Target Gate: one targeted vehicle may occupy the target conflict zone at a time.")

        traci.start(build_sumo_command(args))
        configure_protected_vehicle_types(traci)

        junction_ids = set(traci.junction.getIDList())
        if args.junction_id not in junction_ids:
            raise RuntimeError(f"Junction '{args.junction_id}' was not found in the SUMO network.")

        jx, jy = traci.junction.getPosition(args.junction_id)
        if using_sumo_gui:
            base_engine.refresh_sumo_gui_view(traci, float(jx), float(jy), args.gui_view_radius_m)

        sim_start_time = float(traci.simulation.getTime())
        wall_start_time = time.monotonic()
        status_topic = f"{args.topic_prefix}/status"

        base_engine.publish_json(
            client,
            status_topic,
            {
                "event_type": "scenario1_protected_engine_started",
                "generated_at_utc": utc_now_iso(),
                "junction_id": args.junction_id,
                "junction_x": float(jx),
                "junction_y": float(jy),
                "model": str(args.model),
                "metadata": str(args.metadata),
                "min_risk": args.min_risk,
                "risk_source": args.risk_source,
                "vehicle_groups": args.vehicle_groups,
                "prepare_speed_mps": PREPARE_SPEED_MPS,
                "wait_speed_mps": WAIT_SPEED_MPS,
                "prepare_to_wait_distance_m": PREPARE_TO_WAIT_DISTANCE_M,
                "wait_at_entry_distance_m": WAIT_AT_ENTRY_DISTANCE_M,
                "conflict_entry_guard_distance_m": CONFLICT_ENTRY_GUARD_DISTANCE_M,
                "entry_hold_distance_m": ENTRY_HOLD_DISTANCE_M,
                "entry_creep_speed_mps": ENTRY_CREEP_SPEED_MPS,
                "real_time": bool(args.real_time),
                "realtime_factor": args.realtime_factor,
            },
            retain=True,
        )

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = float(traci.simulation.getTime())

            if using_sumo_gui and step < max(args.gui_refresh_steps, 0):
                base_engine.refresh_sumo_gui_view(traci, float(jx), float(jy), args.gui_view_radius_m)

            active_vehicle_ids = set(map(str, traci.vehicle.getIDList()))
            history_store.prune_missing(active_vehicle_ids)

            current_states: dict[str, dict[str, Any]] = {}
            candidate_histories: dict[str, list[dict[str, Any]]] = {}
            predictions_by_vehicle: dict[str, list[dict[str, Any]]] = {}

            for vehicle_id in active_vehicle_ids:
                state = base_engine.collect_vehicle_state(traci, vehicle_id, sim_time, float(jx), float(jy))
                current_states[vehicle_id] = state
                history_store.add(vehicle_id, state)

                if not base_engine.vehicle_group_allowed(str(state["vehicle_group"]), args.vehicle_groups):
                    continue
                if float(state["distance_to_junction_center_m"]) > float(args.near_radius_m):
                    continue
                if not history_store.is_ready(vehicle_id):
                    continue

                candidate_histories[vehicle_id] = history_store.history(vehicle_id)

            should_predict = args.prediction_interval_steps <= 1 or step % args.prediction_interval_steps == 0
            if should_predict and candidate_histories:
                predictions_by_vehicle = predictor.predict_many(candidate_histories)

            if args.publish_predictions and predictions_by_vehicle:
                for vehicle_id, predictions in predictions_by_vehicle.items():
                    base_engine.publish_json(
                        client,
                        f"{args.topic_prefix}/predictions/{vehicle_id}",
                        {
                            "event_type": "trajectory_prediction",
                            "generated_at_utc": utc_now_iso(),
                            "simulation_time": round(sim_time, 4),
                            "vehicle_id": vehicle_id,
                            "predictions": predictions,
                        },
                    )

            all_alerts = build_pair_alerts(
                predictions_by_vehicle,
                simulation_time=sim_time,
                min_risk_level=args.min_risk,
            )
            protection_candidates = base_engine.sorted_limited_alerts(all_alerts, 0)
            alerts = base_engine.sorted_limited_alerts(all_alerts, args.max_alerts_per_cycle)
            high_alerts = [alert for alert in protection_candidates if str(alert.get("risk_level")) == "HIGH"]
            high_alerts_seen += len(high_alerts)

            update_alert_history(alert_history, protection_candidates, sim_time)
            collision_diagnoses += write_collision_diagnosis_rows(
                traci,
                diagnosis_writer,
                diagnosis_handle,
                sim_time,
                alert_history,
                gate_state,
                float(jx),
                float(jy),
            )

            new_protections, new_releases = maintain_intersection_gate_controls(
                traci,
                protection_writer,
                protection_handle,
                gate_state,
                active_vehicle_ids,
                current_states,
                sim_time,
                float(jx),
                float(jy),
                high_alerts,
            )
            protections_applied += new_protections
            releases += new_releases

            for alert in alerts:
                if args.alert_mode == "episode":
                    episode_update = base_engine.update_episode_state_for_alert(
                        active_episode_states,
                        alert,
                        args.episode_reset_s,
                    )
                    should_publish = episode_update is not None
                    if episode_update is not None:
                        alert.update(episode_update)
                elif args.alert_mode == "cooldown":
                    should_publish = base_engine.should_publish_alert(
                        last_publish_times,
                        alert,
                        args.alert_cooldown_s,
                    )
                    alert.setdefault("episode_status", "COOLDOWN_DETECTION")
                else:
                    should_publish = True
                    alert.setdefault("episode_status", "DETECTION")

                if not should_publish:
                    continue

                total_alerts += 1
                add_mqtt_comparison_fields(alert)
                base_engine.publish_json(client, f"{args.topic_prefix}/alerts", alert)
                base_engine.publish_json(
                    client,
                    f"{args.topic_prefix}/alerts/{risk_topic_suffix(alert['risk_level'])}",
                    alert,
                )
                write_sent_alert_if_available(sent_log_writer, sent_log_handle, alert)

            if args.alert_mode == "episode":
                base_engine.prune_episode_states(active_episode_states, sim_time, args.episode_reset_s)

            if args.real_time:
                base_engine.pace_real_time(sim_time, sim_start_time, wall_start_time, args.realtime_factor)

            if args.status_interval_steps > 0 and step % args.status_interval_steps == 0:
                base_engine.publish_json(
                    client,
                    status_topic,
                    {
                        "event_type": "scenario1_protected_engine_status",
                        "generated_at_utc": utc_now_iso(),
                        "simulation_time": round(sim_time, 4),
                        "step": step,
                        "active_vehicles": len(active_vehicle_ids),
                        "ready_vehicle_buffers": len(predictions_by_vehicle),
                        "alerts_published": total_alerts,
                        "high_alerts_seen": high_alerts_seen,
                        "protections_applied": protections_applied,
                        "releases": releases,
                        "collision_diagnoses": collision_diagnoses,
                        "priority_vehicle_id": gate_state.get("current_priority_vehicle_id", ""),
                        "reservation_queue": gate_state.get("reservation_queue", []),
                        "conflict_zone_vehicle_ids": gate_state.get("conflict_zone_vehicle_ids", []),
                    },
                    retain=True,
                )

            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break

        base_engine.publish_json(
            client,
            status_topic,
            {
                "event_type": "scenario1_protected_engine_stopped",
                "generated_at_utc": utc_now_iso(),
                "steps": step,
                "alerts_published": total_alerts,
                "high_alerts_seen": high_alerts_seen,
                "protections_applied": protections_applied,
                "releases": releases,
                "collision_diagnoses": collision_diagnoses,
                "priority_vehicle_id": gate_state.get("current_priority_vehicle_id", ""),
                "reservation_queue": gate_state.get("reservation_queue", []),
                "conflict_zone_vehicle_ids": gate_state.get("conflict_zone_vehicle_ids", []),
            },
            retain=True,
        )

    except Exception as exc:
        base_engine.publish_json(
            client,
            f"{args.topic_prefix}/status",
            {
                "event_type": "scenario1_protected_engine_error",
                "generated_at_utc": utc_now_iso(),
                "error": str(exc),
            },
            retain=True,
        )
        print(str(exc), file=sys.stderr)
        return 1

    finally:
        try:
            active_vehicle_ids = set(map(str, traci.vehicle.getIDList()))
            for vehicle_id in list(gate_state.get("controlled", {})):
                if vehicle_id in active_vehicle_ids:
                    try:
                        release_gate_vehicle(traci, gate_state, vehicle_id, float(traci.simulation.getTime()))
                    except Exception:
                        pass
            traci.close()
        except Exception:
            pass
        client.loop_stop()
        client.disconnect()
        if sent_log_handle is not None:
            sent_log_handle.close()
        protection_handle.close()
        diagnosis_handle.close()

    print(
        "Protected collision engine finished. "
        f"Steps={step}, alerts_published={total_alerts}, high_alerts_seen={high_alerts_seen}, "
        f"protections_applied={protections_applied}, releases={releases}, "
        f"collision_diagnoses={collision_diagnoses}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
