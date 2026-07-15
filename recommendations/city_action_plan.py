"""Deterministic, in-memory city action-plan core; no ML or provider calls."""

from __future__ import annotations
import hashlib
from collections import defaultdict
from datetime import timedelta

PRIORITY = {
    "critical": 1.0,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
    "monitor_only": 0.1,
}
CATALOG = {
    "DYNAMIC_TOP_1PCT": ("INCREASE_PATROL", "SPEED_MONITORING"),
    "DYNAMIC_TOP_5PCT": ("INCREASE_PATROL", "SPEED_MONITORING"),
    "HOTSPOT_TOP_20": ("ENGINEERING_INSPECTION",),
    "HOTSPOT_TOP_50": ("ENGINEERING_INSPECTION",),
    "SEVERE_WEATHER": ("SPEED_MONITORING", "ROAD_SURFACE_CHECK", "DRIVER_WARNING"),
    "ROAD_REPAIR": ("REPAIR_ZONE_SAFETY_REVIEW",),
    "MAJOR_EVENT": ("EVENT_TRAFFIC_CONTROL", "PEDESTRIAN_FLOW_MONITORING"),
    "HEAVY_TRAFFIC": ("CONGESTION_MONITORING",),
}
WEATHER = {"SPEED_MONITORING", "ROAD_SURFACE_CHECK", "DRIVER_WARNING"}
EVENT = {"EVENT_TRAFFIC_CONTROL", "PEDESTRIAN_FLOW_MONITORING"}
REPAIR = {"REPAIR_ZONE_SAFETY_REVIEW"}
TRAFFIC = {"CONGESTION_MONITORING"}


def generate_city_action_plan(
    segments,
    *,
    batch_id,
    prediction_datetime,
    horizon_hours=24,
    max_actions=10,
    minimum_priority="medium",
):
    rows = list(
        segments.to_dict("records") if hasattr(segments, "to_dict") else segments
    )
    minv = PRIORITY[minimum_priority]
    candidates = []
    for r in rows:
        reasons = set(r.get("reasons", []))
        p = r.get("operational_priority", "low")
        strong = bool(
            reasons
            & {
                "SEVERE_WEATHER",
                "HEAVY_TRAFFIC",
                "ROAD_REPAIR",
                "MAJOR_EVENT",
                "MULTI_SIGNAL_AGREEMENT",
            }
        )
        if PRIORITY.get(p, 0) < minv and not (
            p == "monitor_only" and "PROVIDER_DEGRADED" in reasons
        ):
            continue
        if p == "medium" and not strong:
            continue
        codes = {c for x in reasons for c in CATALOG.get(x, ())} or (
            {"CONTINUE_MONITORING"} if "PROVIDER_DEGRADED" in reasons else set()
        )
        for code in codes:
            candidates.append((code, r, reasons))
    groups = defaultdict(list)
    for code, r, reasons in candidates:
        label = r.get("road_name") or r.get("road_ref") or r["road_segment_id"]
        groups[(code, str(label).strip().casefold())].append((r, reasons))
    actions = []
    for (code, label), items in groups.items():
        rs = [x[0] for x in items]
        reasons = sorted(set().union(*(x[1] for x in items)))
        best = max(rs, key=lambda x: float(x.get("dynamic_percentile", 0)))
        n = len(rs)
        context = max(
            [float(x.get("weather_severity_score", 0) or 0) for x in rs] + [0]
        )
        uncertainty = max(
            (x.get("uncertainty", "low") for x in rs),
            key=lambda x: {"low": 0, "medium": 1, "high": 2}.get(x, 0),
        )
        score = max(
            0,
            min(
                1,
                0.4 * float(best.get("dynamic_percentile", 0))
                + 0.25 * float(best.get("historical_hotspot_percentile", 0))
                + 0.2 * context
                + 0.1 * PRIORITY.get(best.get("operational_priority"), 0)
                + 0.05 * min(n / 5, 1)
                - {"low": 0, "medium": 0.05, "high": 0.1}.get(uncertainty, 0.1),
            ),
        )
        start, end, basis = _period(code, best, prediction_datetime, horizon_hours)
        loc = {
            "display_name": best.get("road_name")
            or best.get("road_ref")
            or "участок дороги",
            "road_name": best.get("road_name"),
            "road_ref": best.get("road_ref"),
            "segment_ids": sorted(str(x["road_segment_id"]) for x in rs),
            "center": {"lon": best.get("lon"), "lat": best.get("lat")},
        }
        aid = hashlib.sha1(
            (batch_id + code + label + "|".join(loc["segment_ids"])).encode()
        ).hexdigest()[:16]
        actions.append(
            {
                "action_id": aid,
                "action_code": code,
                "action_priority_score": score,
                "action_priority": "critical"
                if score >= 0.8
                else "high"
                if score >= 0.6
                else "medium"
                if score >= 0.4
                else "monitor",
                "location": loc,
                "recommended_period": {"start": start, "end": end, "basis": basis},
                "reason_codes": reasons,
                "evidence": {
                    "best_dynamic_rank": best.get("dynamic_rank"),
                    "maximum_dynamic_percentile": best.get("dynamic_percentile"),
                    "best_hotspot_rank": best.get("historical_hotspot_rank"),
                    "maximum_hotspot_percentile": best.get(
                        "historical_hotspot_percentile"
                    ),
                    "supporting_segments": n,
                },
                "text": _text(code, loc["display_name"], start, end),
                "warnings": sorted(set(sum((x.get("warnings", []) for x in rs), []))),
                "requires_human_confirmation": True,
            }
        )
    actions = sorted(
        actions,
        key=lambda x: (-x["action_priority_score"], x["action_code"], x["action_id"]),
    )[:max_actions]
    for i, a in enumerate(actions, 1):
        a["action_rank"] = i
    return {
        "batch_id": batch_id,
        "prediction_datetime": prediction_datetime,
        "horizon_hours": horizon_hours,
        "plan_version": "city_action_plan_v1",
        "summary": {
            "segments_analyzed": len(rows),
            "candidate_segments": len({r[1]["road_segment_id"] for r in candidates}),
            "groups_created": len(groups),
            "actions_returned": len(actions),
        },
        "actions": actions,
    }


def _period(code, r, t, h):
    key = (
        "event"
        if code in EVENT
        else "repair"
        if code in REPAIR
        else "weather"
        if code in WEATHER
        else "traffic"
        if code in TRAFFIC
        else None
    )
    if key:
        return (
            r.get(f"{key}_start") or r.get(f"{key}_worst_period_start") or t,
            r.get(f"{key}_end") or r.get(f"{key}_worst_period_end") or t,
            "%s_period" % key,
        )
    return (
        t,
        (
            __import__("datetime").datetime.fromisoformat(t.replace("Z", "+00:00"))
            + timedelta(hours=h)
        ).isoformat(),
        "prediction_horizon",
    )


def _text(code, location, start, end):
    en = f"Consider {code.lower().replace('_', ' ')} for {location} during the recommended period."
    return {
        "ru": f"Рассмотреть действие {code} на участке {location} в рекомендуемый период.",
        "kz": f"Ұсынылатын кезеңде {location} учаскесінде {code} әрекетін қарастыру.",
        "en": en,
    }
